
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name undistorted_center \
    --dataset oxnard \
    --data_path=./data/datasets/triplets_daytime_undistorted_center \
    --height 416 \
    --width 608 \
    --batch_size 4 \
    --num_epochs 200 \
    --log_dir ./output
