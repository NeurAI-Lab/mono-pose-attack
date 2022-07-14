import numpy as np
import os

global_poses_folder = "./gt_poses_global"
local_poses_folder = "./gt_poses_local"
for pose_file in os.listdir(global_poses_folder):
    gt_poses_path = os.path.join(global_poses_folder, pose_file)
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        pose = np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i]))
        pose = pose.reshape(1,16)
        gt_local_poses.append(pose.squeeze())

    output_path = os.path.join(local_poses_folder, pose_file)
    np.savetxt(output_path, gt_local_poses, delimiter=' ', fmt='%1.8e')