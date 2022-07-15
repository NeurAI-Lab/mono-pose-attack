# Adversarial Attacks on Monocular Pose Estimation

This is the official code for the IROS 2022 paper: [Adversarial Attacks on Monocular Pose Estimation](https://arxiv.org/abs/2207.07032) by [Hemang Chawla](https://scholar.google.com/citations?user=_58RpMgAAAAJ&hl=en&oi=ao), [Arnav Varma](https://scholar.google.com/citations?user=3QSih2AAAAAJ&hl=en&oi=ao), [Elahe Arani](https://www.linkedin.com/in/elahe-arani-630870b2/) and [Bahram Zonooz](https://scholar.google.com/citations?hl=en&user=FZmIlY8AAAAJ).

This codebase implements the adversarial attacks on monocular pose estimation using [SC-Depth](https://github.com/JiawangBian/SC-SfMLearner-Release) as an example repo.

## Setup

Setup the conda environment using:
```bash
conda env install --name env_adversarial_attacks_pose --file requirements.yml 
```

## Dataset for adversarial attack

    For KITTI Raw dataset, download the dataset using this script http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website.
    For KITTI Odometry dataset, download the dataset with color images.

After downloading KITTI dataset, please run:

```bash
DATASET=path_to_dataset
OUTPUT=path_to_output
STATIC_FILES=data/static_frames.txt
python data/prepare_test_data.py $DATASET --dataset-format 'kitti_raw' --dump-root $OUTPUT --width 832 --height 256 --num-threads 12 --static-frames $STATIC_FILES --with-depth
```

Download `pose_test_set.tar.xz` from [drive](https://drive.google.com/file/d/1DTWITOCTOZl6Etzinv8MDk4c0iQHS1mg/view?usp=sharing).
Then run:

```bash
tar xf pose_test_set.tar.xz
```

## Pretrained Models used for adversarial attacks

We use models from [SC-Depth Models](https://1drv.ms/u/s!AiV6XqkxJHE2kxX_Gek5fEQvMGma?e=ZfrnbR) under `resnet50_pose_256`

## Attacks

We demonstrate untargeted and targeted attack on the pose estimation. 
We also measure the impact of cross task attacks. 
Accordingly, the following files can be used.   

| Atack        | Eval  | Filename                              |
|--------------|-------|---------------------------------------|
| Untargeted   | Pose  | `pgd_attack_eval_depth.py`            |
|              | Depth | `pgd_attack_eval_pose.py`             |
| Target Pose  | Pose  | `vo_targeted_attack_eval_vo.py`       |
|              | Depth | `vo_targeted_attack_eval_depth.py`    |
| Target Depth | Pose  | `depth_targeted_attack_eval_vo.py`    |
|              | Depth | `depth_targeted_attack_eval_depth.py` |


### Untargeted Attack

Example code to run untargeted attack on SC-Depth ckpt for KITTI Sequence 09 are given hereafter:

#### Evaluate Pose
```bash
python pgd_attack_eval_vo.py --sequence 09 --pretrained-posenet path_to_posenet_ckpt --pretrained-dispnet path_to_dispnet_ckpt --dataset-dir path_to_odom_dataset --output-dir path_to_output_dir --save-imgs
```

#### Evaluate Depth
```bash
python pgd_attack_eval_depth.py  --sequence 09 --pretrained-posenet path_to_posenet_ckpt --pretrained-dispnet path_to_dispnet_ckpt  --dataset-dir path_to_odom_dataset --gt-dir path_to_depth_gt --output-dir path_to_output_dir --save-imgs
```

### Targeted Attack on Pose
Example code to run targeted attack (move backwards) on SC-Depth ckpt for KITTI Sequence 09 are given hereafter:

#### Evaluate Pose
```bash
python vo_targeted_attack_eval_vo.py --sequence 09 --pretrained-posenet path_to_posenet_ckpt --pretrained-dispnet path_to_dispnet_ckpt --dataset-dir path_to_odom_dataset --output-dir path_to_output_dir --save-imgs --target-mode move_backwards
```

#### Evaluate Depth
```bash
python vo_targeted_attack_eval_vo.py --sequence 09 --pretrained-posenet path_to_posenet_ckpt --pretrained-dispnet path_to_dispnet_ckpt --dataset-dir path_to_odom_dataset --gt-dir path_to_depth_gt --output-dir path_to_output_dir --save-imgs --target-mode move_backwards
```

### Targeted Attack on Depth
Example code to run targeted attack (flip vertical) on SC-Depth ckpt for KITTI Sequence 09 are given hereafter:

#### Evaluate Pose
```bash
python depth_targeted_attack_eval_vo.py --sequence 09 --pretrained-posenet path_to_posenet_ckpt --pretrained-dispnet path_to_dispnet_ckpt --dataset-dir path_to_odom_dataset --output-dir path_to_output_dir --save-imgs --target-mode v
```

#### Evaluate Depth
```bash
python depth_targeted_attack_eval_depth.py --sequence 09 --pretrained-posenet path_to_posenet_ckpt --pretrained-dispnet path_to_dispnet_ckpt --dataset-dir path_to_odom_dataset --gt-dir path_to_depth_gt --output-dir path_to_output_dir --save-imgs --target-mode v
```

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:

<pre>
@inproceedings{chawlavarma2022adversarial,
	author={H. {Chawla} and A. {Varma} and E. {Arani} and B. {Zonooz}},
	booktitle={2022 IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS)},
	title={Adversarial Attacks on Monocular Pose Estimation},
	location={Kyoto, Japan},
	publisher={IEEE (in press)},
	year={2022}
}
</pre>

## License

This project is licensed under the terms of the MIT license.
