# Copyright © NavInfo Europe 2022.

from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import custom_transforms
import torch.utils.data
from inverse_warp import *
from kitti_eval.kitti_odometry import KittiEvalOdom
from PIL import Image
import models
import random
import os
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--save-imgs", action='store_true', help="To save adv imgs")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", required=True, type=str, help="Dataset directory")
parser.add_argument("--output-dir", required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument("--stats-fname", help="expt_name", type=str, default="PGD")
parser.add_argument("--num-workers", type=int, help="number of dataloader workers", default=12)
parser.add_argument('--resnet-layers', type=int, default=50, choices=[18, 50], help='depth network architecture.')
parser.add_argument("--sequence", default='09', type=str, help="sequence to test", choices=['09', '10'])
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument("--target-mode", choices=['invert', 'flip_yaw', 'move_backwards'], type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(torch.utils.data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, seq='09', sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/seq + '.txt'
        self.scenes = [self.root/folder.strip() for folder in open(scene_list_path) if len(folder.strip()) > 0]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)


class PGDAttack:
    def __init__(self,
                 target_mode,
                 data_path,
                 pose_model_pth,
                 depth_model_pth,
                 sequence,
                 eval_out_dir,
                 no_resize,
                 height=256,
                 width=832,
                 img_exts="PNG",
                 save_adv_imgs=False,
                 min_depth=0.1,
                 max_depth=80.0,
                 resnet_layers=50,
                 w1=1,
                 w2=1,
                 w3=1
                 ):

        self.target_mode = target_mode
        self.data_path = data_path
        self.eval_split = sequence
        self.sequence_id = self.eval_split.split("_")[-1]
        self.eval_out_dir = eval_out_dir
        self.save_adv_imgs = save_adv_imgs
        self.img_exts = img_exts
        self.no_resize = no_resize

        self.height = height
        self.width = width

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.device = torch.device("cuda")

        self.resnet_layers = resnet_layers
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.arccos_min = torch.tensor(-1).to(device)
        self.arccos_max = torch.tensor(1).to(device)

        output_dir = Path(self.eval_out_dir)
        output_dir.makedirs_p()

        self.eval_tool = KittiEvalOdom()
        self.gt_dir = "./kitti_eval/gt_poses/"
        print("gt path", self.gt_dir)

        normalize = custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        test_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

        # SC-SfM structure means we need to keep train=True
        self.test_set = SequenceFolder(
            data_path,
            transform=test_transform,
            seq=self.sequence_id,
            sequence_length=3,
            dataset='kitti'
        )

        print("data_path:", data_path)

        weights_pose = torch.load(pose_model_pth)
        self.pose_net = models.PoseResNet().to(device)
        self.pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
        self.pose_net.eval()

        weights = torch.load(depth_model_pth)
        self.disp_net = models.DispResNet(self.resnet_layers, False).to(device)
        self.disp_net.load_state_dict(weights['state_dict'])
        self.disp_net.eval()

        self.models = {}
        self.ivt = [
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda(),
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
        ]

    def process(self, epsilon, num_workers=12):

        dataloader = torch.utils.data.DataLoader(
            self.test_set, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        self.save_dir = os.path.join(self.eval_out_dir, "targeted_" + self.target_mode, "adv_" + str(epsilon))
        os.makedirs(os.path.join(self.save_dir, self.eval_split), exist_ok=True)

        print("save dir: ", self.save_dir)

        self.results_dir = os.path.join(self.save_dir, self.eval_split)
        os.makedirs(self.results_dir, exist_ok=True)

        if self.save_adv_imgs:
            self.adv_dir = os.path.join(self.results_dir, "adv_examples")
            self.noise_dir = os.path.join(self.results_dir, "noise")
            os.makedirs(self.adv_dir, exist_ok=True)
            os.makedirs(self.noise_dir, exist_ok=True)

        self.evaluate(dataloader, self.results_dir, epsilon=epsilon)

    def evaluate(self, dataloader, results_dir, epsilon):
        """Evaluates a pretrained model using a specified test set
        """

        num_iters = min(epsilon + 4, math.ceil(1.25 * epsilon))
        num_iters = int(np.max([np.ceil(num_iters), 1]))

        self.disp_net.eval()
        self.pose_net.eval()

        print("-> Computing predictions with size {}x{}".format(
            self.width, self.height))

        print("len dataloader: ", len(dataloader))

        global_pose = np.eye(4)
        poses = [global_pose[0:3, :].reshape(1, 12)]

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            tgt_img, ref_imgs, _, _ = data
            ref_img1, ref_img2 = ref_imgs

            if i == 0:
                img1, img2 = self.targeted_attack(ref_img1, tgt_img, eps=epsilon, num_iters=num_iters, visualize=False,
                                                   save_img=self.save_adv_imgs, im_num=i)
                with torch.no_grad():
                    pose = self.pose_net(img1, img2)
                    pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
                    pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])

                    global_pose = global_pose @ np.linalg.inv(pose_mat)
                    poses.append(global_pose[0:3, :].reshape(1, 12))

            img1, img2 = self.targeted_attack(tgt_img, ref_img2, eps=epsilon, num_iters=num_iters, visualize=False,
                                              save_img=self.save_adv_imgs, im_num=i+1)

            with torch.no_grad():
                pose = self.pose_net(img1, img2)
                pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
                pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])

                global_pose = global_pose @ np.linalg.inv(pose_mat)
                poses.append(global_pose[0:3, :].reshape(1, 12))

            if i == len(dataloader) - 1:
                if self.save_adv_imgs:
                    save_adv_name = os.path.join(self.adv_dir, str(i + 2) + ".png")
                    save_noise_name = os.path.join(self.noise_dir, str(i + 2) + ".png")
                    save_adv_name_npy = os.path.join(self.adv_dir, str(i + 2) + ".npy")
                    Image.fromarray(
                        np.transpose(255 * (img2 * self.ivt[1] + self.ivt[
                            0]).detach().cpu().squeeze().numpy(),
                                     (1, 2, 0)).astype(np.uint8)
                    ).save(save_adv_name)
                    Image.fromarray(
                        np.transpose(
                            ((img2 * self.ivt[1] + self.ivt[0]).detach().cpu().squeeze().numpy() -
                             (ref_img2.to(device) * self.ivt[1] + self.ivt[
                                 0]).cpu().squeeze().numpy()) * 255.0,
                            (1, 2, 0)
                        ).astype(np.uint8)
                    ).save(save_noise_name)
                    np.save(save_adv_name_npy,
                            (img2 * self.ivt[1] + self.ivt[0]).detach().cpu().numpy()
                            )

        print("len poses:", len(poses))
        poses = np.concatenate(poses, axis=0)
        filename = os.path.join(results_dir, self.eval_split+".txt")
        np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
        self.eval_tool.eval(
            self.gt_dir,
            results_dir,
            alignment='7dof'
        )

    def compute_depth(self, tgt_img, ref_imgs):
        tgt_depth = [1 / disp for disp in self.disp_net(tgt_img)]
        if len(tgt_depth[0].shape) == 3:
            tgt_depth = [depth.unsqueeze(1) for depth in tgt_depth]

        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp for disp in self.disp_net(ref_img)]
            if len(ref_depth[0].shape) == 3:
                ref_depth = [depth.unsqueeze(1) for depth in ref_depth]

            ref_depths.append(ref_depth)

        return tgt_depth, ref_depths

    def compute_pose_with_inv(self, tgt_img, ref_imgs):
        poses = []
        poses_inv = []
        for ref_img in ref_imgs:
            poses.append(self.pose_net(tgt_img, ref_img))
            poses_inv.append(self.pose_net(ref_img, tgt_img))

        return poses, poses_inv

    def pose_vec2mat(self, translation, rot, rotation_mode='euler'):
        """
        Convert 6DoF parameters to transformation matrix.
        Args:s
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 3, 4]
        """
        if rotation_mode == 'euler':
            rot_mat = euler2mat(rot)  # [B, 3, 3]
        elif rotation_mode == 'quat':
            rot_mat = quat2mat(rot)  # [B, 3, 3]
        transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
        return transform_mat

    def generate_target(self, translation, rot):

        if self.target_mode == 'invert':
            target_mat = self.pose_vec2mat(translation, rot).squeeze(0)
            target_mat = torch.vstack((target_mat, torch.tensor([0, 0, 0, 1]).to(device)))
            target_mat = torch.inverse(target_mat)
        elif self.target_mode == 'flip_yaw':
            # yaw is the second one (about y axis)
            rot[:, 1] = -1 * rot[:, 1]
            target_mat = self.pose_vec2mat(translation, rot).squeeze(0)
            target_mat = torch.vstack((target_mat, torch.tensor([0, 0, 0, 1]).to(device)))
        elif self.target_mode == 'move_backwards':
            # z (along the camera) is the third
            translation[:, 2] = -1 * translation[:, 2]
            target_mat = self.pose_vec2mat(translation, rot).squeeze(0)
            target_mat = torch.vstack((target_mat, torch.tensor([0, 0, 0, 1]).to(device)))

        return target_mat

    def rotation_error(self, pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            rot_error (float): rotation error
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = torch.arccos(torch.max(torch.min(d, self.arccos_max), self.arccos_min))
        return rot_error

    def translation_error(self, pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            trans_error (float): translation error
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return trans_error

    def process_pose_inputs(self, img1, img2, target_mat):
        """Pass a minibatch through the network and generate images and losses
        """
        img1 = img1.to(device)
        img2 = img2.to(device)

        # compute output
        pose = self.pose_net(img1, img2)
        translation = pose[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = pose[:, 3:]
        pose_mat = self.pose_vec2mat(translation, rot).squeeze(0)
        pose_mat = torch.vstack((pose_mat, torch.tensor([0, 0, 0, 1]).to(device)))

        pose_error = torch.matmul(
            torch.inverse(pose_mat),
            target_mat
        )

        r_err = self.rotation_error(pose_error)
        t_err = self.translation_error(pose_error)

        if self.target_mode == 'invert':
            loss = r_err + t_err
        elif self.target_mode == 'flip_yaw':
            loss = r_err
        elif self.target_mode == 'move_backwards':
            loss = t_err

        return loss

    def targeted_attack(self, img1, img2, eps, num_iters, alpha=1,
                         visualize=False, save_img=False, using_noise=True,
                         im_num=None):

        img1 = img1.to(device)
        img2 = img2.to(device)

        if save_img:
            save_adv_name = os.path.join(self.adv_dir, str(im_num) + ".png")
            save_noise_name = os.path.join(self.noise_dir, str(im_num) + ".png")
            save_adv_name_npy = os.path.join(self.adv_dir, str(im_num) + ".npy")

        if eps == 0:
            if save_img:
                Image.fromarray(
                    np.transpose(255 * (img1 * self.ivt[1] + self.ivt[
                        0]).detach().cpu().squeeze().numpy(),
                                 (1, 2, 0)).astype(np.uint8)
                ).save(save_adv_name)
                Image.fromarray(
                    np.transpose(
                        ((img1 * self.ivt[1] + self.ivt[0]).detach().cpu().squeeze().numpy() -
                         (img1 * self.ivt[1] + self.ivt[0]).cpu().squeeze().numpy()) * 255.0, (1, 2, 0)
                    ).astype(np.uint8)
                ).save(save_noise_name)
                np.save(save_adv_name_npy,
                        (img1 * self.ivt[1] + self.ivt[0]).detach().cpu().numpy()
                        )
            return img1, img2

        eps /= 255.0
        eps_pose = torch.ones_like(img1.to(device)) * eps / self.ivt[1]

        alpha /= 255.0

        alpha_pose = alpha / self.ivt[1]
        alpha_pose = alpha_pose.view(1, 3, 1, 1).to(device)

        adv_img1 = img1.clone().to(device)
        adv_img2 = img2.clone().to(device)

        ub_max_pose = (torch.ones_like(adv_img1) - self.ivt[0]) / self.ivt[1]
        lb_min_pose = (torch.zeros_like(adv_img1) - self.ivt[0]) / self.ivt[1]

        ub_pose_1 = torch.min(adv_img1 + eps_pose, ub_max_pose)
        lb_pose_1 = torch.max(adv_img1 - eps_pose, lb_min_pose)
        ub_pose_2 = torch.min(adv_img2 + eps_pose, ub_max_pose)
        lb_pose_2 = torch.max(adv_img2 - eps_pose, lb_min_pose)

        if using_noise:
            adv_img1 = adv_img1 + \
                                   torch.FloatTensor(adv_img1.size()).uniform_(-eps, eps).cuda()
            adv_img1 = torch.max(torch.min(adv_img1, ub_pose_1), lb_pose_1)
            adv_img2 = adv_img2 + \
                           torch.FloatTensor(adv_img2.size()).uniform_(-eps, eps).cuda()
            adv_img2 = torch.max(torch.min(adv_img2, ub_pose_2), lb_pose_2)
        del ub_max_pose, lb_min_pose, eps_pose

        if visualize:
            plt.ion()
            plt.show()

        # generate target
        pose = self.pose_net(img1, img2)
        pose = pose.detach()
        translation = pose[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = pose[:, 3:]
        target_mat = self.generate_target(translation, rot)

        for i in range(num_iters):

            adv_img1.requires_grad = True
            adv_img2.requires_grad = True

            loss = self.process_pose_inputs(adv_img1, adv_img2, target_mat)
            loss.backward()

            noise_img1 = alpha_pose * torch.sign(adv_img1.grad)
            noise_img2 = alpha_pose * torch.sign(adv_img2.grad)

            adv_img1 = adv_img1.detach() + noise_img1
            adv_img1 = torch.max(torch.min(adv_img1, ub_pose_1), lb_pose_1)
            adv_img2 = adv_img2.detach() + noise_img2
            adv_img2 = torch.max(torch.min(adv_img2, ub_pose_2), lb_pose_2)

            if (i == num_iters - 1) and (visualize or save_img):
                if visualize:
                    plt.imshow(
                        np.transpose(
                            (adv_img1 * self.ivt[1] + self.ivt[0]).detach().cpu().squeeze().numpy() *
                            255.0, (1, 2, 0)).astype(np.uint8)
                    )
                    plt.pause(1)
                if save_img:
                    Image.fromarray(
                        np.transpose(255 * (adv_img1 * self.ivt[1] + self.ivt[
                            0]).detach().cpu().squeeze().numpy(),
                                     (1, 2, 0)).astype(np.uint8)
                    ).save(save_adv_name)
                    Image.fromarray(
                        np.transpose(
                            ((adv_img1 * self.ivt[1] + self.ivt[0]).detach().cpu().squeeze().numpy() -
                             (img1 * self.ivt[1] + self.ivt[0]).cpu().squeeze().numpy()) * 255.0, (1, 2, 0)
                        ).astype(np.uint8)
                    ).save(save_noise_name)
                    np.save(save_adv_name_npy,
                            (adv_img1 * self.ivt[1] + self.ivt[0]).detach().cpu().numpy()
                            )

        return adv_img1.detach(), adv_img2.detach()


if __name__ == '__main__':
    args = parser.parse_args()
    stats_all = []

    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    attack = PGDAttack(
        target_mode=args.target_mode,
        pose_model_pth=args.pretrained_posenet,
        depth_model_pth=args.pretrained_dispnet,
        data_path=args.dataset_dir,
        sequence=args.sequence,
        eval_out_dir=args.output_dir,
        height=args.img_height,
        width=args.img_width,
        img_exts=args.img_exts,
        save_adv_imgs=args.save_imgs,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        no_resize=args.no_resize,
        resnet_layers=args.resnet_layers,
        w1=w1,
        w2=w2,
        w3=w3
    )

    epsilons = [0, 1, 2, 4]

    for epsilon in epsilons:
        attack.process(epsilon=epsilon, num_workers=args.num_workers)
