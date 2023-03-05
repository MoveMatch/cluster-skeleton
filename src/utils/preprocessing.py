import yaml
import argparse
import torch
import boto3
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from boto.s3.connection import S3Connection


# Create a dataset class that reads the files from the S3 bucket
class S3Dataset(Dataset):
    def __init__(self, s3, bucket, prefix):
        self.s3 = s3
        self.bucket = bucket
        self.prefix = prefix
        self.filenames = self.get_filenames()
        self.transform = transforms.Compose([
            # make this a param later
            transforms.Resize((50,50)),
            transforms.ToTensor()
        ])
        
    def get_filenames(self):
        filenames = []
        
        # Set up the parameters for the list_objects call
        params = {
            'Bucket': self.bucket,
            'Prefix': self.prefix
        }

        # Initiate S3 connection
        #conn = S3Connection(KEY_PAIR_HERE)

        # Get the bucket
        # bucket = conn.get_bucket(BUCKET_NAME_HERE)

        # sub bucket (optional)
        # prefix = "subfolder1/subfolder2/
        for key in bucket.list(prefix=prefix):
            filenames.append(key.name)
            
        return filenames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Use the Amazon S3 client to download the file from the bucket
        print(self.filenames[idx])
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.filenames[idx])
        image_data = obj['Body']
        
        img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), 1)

        # Convert the NumPy array to a PIL image
        image = Image.fromarray(img)

        # tensor and resize
        image = self.transform(image)

        print("RESIZED: ", image.shape)

        # Extract the class label from the filename and return it with the image tensor
        label = self.filenames[idx].split('/')[0]
        #print(image.shape)
        return image, label