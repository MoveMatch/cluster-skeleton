# Activate the conda environment
# source activate myenv

# Set up logging
mkdir -p logs/train
LOG_FILE=logs/train/`date +%Y-%m-%d_%H-%M-%S`.log

# Run the training script
python src/train.py --config config/train_config.yml --s3-bucket my-bucket --s3-prefix data 2>&1 | tee $LOG_FILE

# Set up training in TensorBoard
tensorboard --logdir logs/train --port 6006 &