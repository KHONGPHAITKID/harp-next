CUDA_VISIBLE_DEVICES=6 python main.py --dataset nuscenes --path_dataset ../dataset/nuscenes/ --mainconfig ./configs/main/main-config.yaml --netconfig ./configs/net/harpnext-nuscenes-tinyvim.yaml --log_path ./logs/harpnext_tinyvim-nuscenes-64x512-trainval --gpu 0 --seed 0 --fp16 --trainval

CUDA_VISIBLE_DEVICES=6 python main.py --dataset nuscenes --path_dataset ../dataset/nuscenes/ --mainconfig ./configs/main/main-config.yaml --netconfig ./configs/net/harpnext-nuscenes-convmonarch.yaml --log_path ./logs/harpnext_convmonarch-nuscenes-64x512-train --gpu 0 --seed 0 --fp16

CUDA_VISIBLE_DEVICES=1 python main.py --dataset semantic_kitti --path_dataset ../dataset/SemanticKitti/data_odometry_velodyne --mainconfig ./configs/main/main-config.yaml --netconfig ./configs/net/harpnext-semantickitti-convmonarch.yaml --log_path ./logs/harpnext-semantickitti-64x512-train --gpu 0 --seed 0 --fp16
