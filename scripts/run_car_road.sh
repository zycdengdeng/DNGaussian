dataset=$1
workspace=$2
export CUDA_VISIBLE_DEVICES=$3
# Optional: vehicle calibration folder, transform JSON, and timestamp for vehicle rendering
vehicle_calib=${4:-""}
transform_json=${5:-""}
timestamp=${6:-""}


python train_llff.py  -s $dataset --model_path $workspace -r 1 --eval --n_sparse 3  --iterations 6000 --lambda_dssim 0.2 \
            --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
            --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
            --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
            --scaling_lr 0.003 \
            --shape_pena 0.002 --opa_pena 0.001 \
            --near 0


python render.py -s $dataset --model_path $workspace -r 1
python metrics.py --model_path $workspace

# Vehicle camera rendering (if calibration info provided)
if [ -n "$vehicle_calib" ] && [ -n "$transform_json" ] && [ -n "$timestamp" ]; then
    echo "Rendering vehicle camera viewpoints..."
    python render_vehicle.py --model_path $workspace \
        --vehicle_calib $vehicle_calib \
        --transform_json $transform_json \
        --timestamp $timestamp \
        --render_scale 4
fi
