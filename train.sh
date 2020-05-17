$PIPELINE_CONFIG_PATH=PROJECT_DIR/config/prod_det_pipeline.config
$MODEL_DIR=PROJECT_DIR/model_logs
$NUM_TRAIN_STEPS=50000
$python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --alsologtostderr