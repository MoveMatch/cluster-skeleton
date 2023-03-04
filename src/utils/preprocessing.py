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

# data is located in s3://movematch/CatsVsDogsDataset/ in folders for each class

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

        conn = S3Connection('AKIATYYGSIDS7FFX3FXH','6WWIEeyU/WjMgvIStKMqmlZ7iHmNS/WMHW21fNV1')
        bucket = conn.get_bucket('movematch')
        prefix = "CatsVsDogsDataset/"
        for key in bucket.list(prefix=prefix):
            filenames.append(key.name)
            #print((key.name))

        print(self.bucket)
        print(self.prefix)
        
        # Use the list_objects method to get a list of all the objects in the bucket
        """        while True:
            response = self.s3.list_objects(**params)
            for obj in response['Contents']:
                filenames.append(obj['Key'])
                
            # Check if there are more objects to list
            if 'NextMarker' in response:
                params['Marker'] = response['NextMarker']
            else:
                break

        print(filenames[0])"""
        print(len(filenames))
            
        return filenames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Use the Amazon S3 client to download the file from the bucket

        print(self.filenames[idx])
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.filenames[idx])
        #print(obj)
        image_data = obj['Body']
        
        img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), 1)
        #print(img.shape)
        #cv2.imwrite("test.jpg", img)
        # Use PyTorch's built-in image processing functions to convert the image data to a tensor
        #image = np.asarray(bytearray(image_data), dtype=np.uint8)
        #print(image.shape)
        
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(img)

        # tensor and resize
        image = self.transform(image)

        print("RESIZED: ", image.shape)

        #image = img

        # Extract the class label from the filename and return it with the image tensor
        label = self.filenames[idx].split('/')[0]
        #print(image.shape)
        return image, label

# Create an instance of the dataset and use it to create a data loader
#dataset = S3Dataset(s3, bucket)
#data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the data loader to process the images and labels
#for batch in data_loader:
#    images, labels = batch
#    print(images.shape, labels)