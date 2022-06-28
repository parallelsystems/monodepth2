
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name mono_model_undistort2 \
    --dataset oxnard \
    --data_path=./data/datasets/triplets_daytime_undistorted2 \
    --height 416 \
    --width 608 \
    --batch_size 4 \
    --num_epochs 200 \
    --log_dir ./output
