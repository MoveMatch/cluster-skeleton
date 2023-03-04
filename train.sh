# Activate the conda environment
#conda init
#conda activate pytorch

PATH_TO_S3="movematch"
PREFIX="CatsVsDogsDataset"
PATH_TO_CONFIG="./config/train_config.yml"

# Set up logging
mkdir -p logs/train
LOG_FILE=logs/train/`date +%Y-%m-%d_%H-%M-%S`.log
echo "logging to $LOG_FILE"

# Run the training script
python src/train.py --config $PATH_TO_CONFIG --s3-bucket $PATH_TO_S3 --prefix $PREFIX | tee $LOG_FILE

# Set up training in TensorBoard
#tensorboard --logdir logs/train --port 6006 &