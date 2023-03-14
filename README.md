# Test-Train Branch

This branch is an example of how the training pipeline should work. It contains the correct s3 connections, the working training script, and pulls the correct data. 

## Requirements

Make sure you have aws cli installed and configured. Smoke test: run aws s3 ls and see if you get a list of buckets. For help with configuring aws cli, message in the discord or check confluence docs.

Also make sure you have a virtual environment or conda environment with the packages in requirements.txt installed.

You may require to brew install s3cmd.

## Usage

To get a training run going, you just need to run the train.sh script. It will pull the data from s3, train the model, and save the model checkpoints.

The script requires 3 arguments: 

```bash
PATH_TO_S3="PARENT_BUCKET_NAME"
PREFIX="SUB_BUCKET_NAME"
PATH_TO_CONFIG="PATH_TO_TRAINING_CONFIG_FILE"
```

Currently, a working example is in the train.sh which can be used for reference. 

Any hyperparameters you want to change can be changed in the config file.

and then you can run the script like so:
./train.sh

## Updating

Changes to model architecture can be made in the src/model.py file

Changes to training loop can be made in the src/train.py file

Changes to the s3 dataloader can be made in the src/dataloader.py file


## License

[MIT](https://choosealicense.com/licenses/mit/)