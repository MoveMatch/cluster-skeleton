import yaml
import argparse
import torch
import boto3
from utils.preprocessing import S3Dataset
from torch.utils.data import Dataset, DataLoader
import cv2
from boto.s3.connection import S3Connection
from dataloader import MotionMistakeDataset
from dotenv import load_dotenv
import os

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, help="Path to the model config file")
parser.add_argument('--s3_bucket', required=True, help='Name of the S3 bucket where the data is stored')
parser.add_argument('--prefix', required=True, help="Sub-Directory within movematch bucket")
args = parser.parse_args()

# setups
load_dotenv()
s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")

# Load the model config file
with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

dataset = MotionMistakeDataset(
  f"pipe:aws s3 cp --quiet s3://{args.s3_bucket}/{args.prefix}/0.tar -", mistake_horizon=20
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,
    pin_memory=True
)

# Make sure the data is loaded correctly
print(dataloader)
dataiter = iter(dataloader)
batch = dataiter.next()
print(batch)

sample = batch.keys()
print(sample)
print(batch['sentences'])



# Extract the model configuration parameters
hidden_units = config["model"]["hidden_units"]
learning_rate = config["model"]["learning_rate"]
optimizer = config["model"]["optimizer"]
loss_function = config["model"]["loss_function"]
metrics = config["model"]["metrics"]

# Extract the training configuration parameters
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
validation_split = config["training"]["validation_split"]

# Extract the inference configuration parameters
threshold = config["inference"]["threshold"]


# Build the model
model = torch.nn.Sequential(
    torch.nn.Flatten(input_shape=(1, 28, 28)),
    *[torch.nn.Linear(28*28, units) for units in hidden_units],
    torch.nn.Linear(hidden_units[-1], 10),
    torch.nn.LogSoftmax(dim=-1)
)

# Compile the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.NLLLoss()

# Train the model
for epoch in range(epochs):
    # Perform one epoch of training
    model.train()
    for x_batch, y_batch in zip(x_train, y_train):
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        test_loss = loss_fn(logits, y_test)
        test_acc = (logits.argmax(dim=-1) == y_test).float().mean()
        #print(f"Epoch {epoch+1} - Test loss: {test_loss:.3f} - Test accuracy: {test_acc:.3f}")

# Save the model
torch.save(model.state_dict(), "model.pt")