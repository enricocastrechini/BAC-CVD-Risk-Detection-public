# Copyright 2024 Enrico Castrechini
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description: [Brief description of what the file or class does]

from src.data.breast_segment import BreastSegmenter

import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

class MakeRGB(object):
    def __call__(self, img):
        img = np.asarray(img)
        img = Image.fromarray(img)
        return img.convert('RGB')

class MaskCropping(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def determine_aspect_ratio(self, width, height):
        return width / height

    def decide_crop_orientation(self, original_aspect_ratio, target_aspect_ratio):
        if original_aspect_ratio > target_aspect_ratio:
            return "horizontal"
        else:
            return "vertical"

    def crop_image(self, image, orientation, target_width, target_height):
        original_height, original_width = image.shape[:2]
        if orientation == "horizontal":
            new_width = int(original_height * (target_width / target_height))
            left = (original_width - new_width) // 2
            cropped_image = image[:, left:left+new_width]
        else:
            new_height = int(original_width / (target_width / target_height))
            top = (original_height - new_height) // 2
            cropped_image = image[top:top+new_height, :]
        return cropped_image

    def resize_image(self, image, target_width, target_height):
        return cv2.resize(image, (target_width, target_height))

    def apply_letterboxing(self, image, target_width, target_height):
        original_height, original_width = image.shape[:2]
        if original_width / original_height > target_width / target_height:
            # Add vertical letterboxing
            pad_size = int((original_width / original_height * target_height - target_width) / 2)
            padded_image = cv2.copyMakeBorder(image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            # Add horizontal letterboxing
            pad_size = int((original_height / original_width * target_width - target_height) / 2)
            padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image

    def __call__(self, image):
        # Split the image into channels
        image = np.asarray(image)
        channels = cv2.split(image)
        masked_cropped_channels = []

        for i, channel in enumerate(channels):
            # Segment the mammogram for each channel separately
            segmenter = BreastSegmenter()
            mask, bbox = segmenter(channel)

            masked_channel = channel.copy()
            masked_channel[~mask] = 0  # Set pixels outside the mask to zero (black)

            # Apply bounding box to crop the channel
            cropped_channel = masked_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            # Resize the cropped channel to the fixed size
            resized_channel = self.resize_image(cropped_channel, self.output_size[0], self.output_size[1])

            # Integrate the cropping and resizing algorithm
            original_aspect_ratio = self.determine_aspect_ratio(cropped_channel.shape[1], cropped_channel.shape[0])
            target_aspect_ratio = self.determine_aspect_ratio(self.output_size[0], self.output_size[1])
            crop_orientation = self.decide_crop_orientation(original_aspect_ratio, target_aspect_ratio)
            cropped_resized_channel = self.crop_image(resized_channel, crop_orientation, self.output_size[0], self.output_size[1])

            # Apply letterboxing if necessary
            final_channel = self.apply_letterboxing(cropped_resized_channel, self.output_size[0], self.output_size[1])

            # Append the processed channel to the list
            masked_cropped_channels.append(final_channel)

        # Merge the cropped channels back into a three-channel image
        masked_cropped_image = cv2.merge(masked_cropped_channels)

        return masked_cropped_image
    

# class MaskCropping(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, image):
#         # Split the image into channels
#         image = np.asarray(image)
#         channels = cv2.split(image)
#         masked_cropped_channels = []
        

#         for i, channel in enumerate(channels):
#             # Segment the mammogram for each channel separately
#             segmenter = BreastSegmenter()
#             mask, bbox = segmenter(channel)

#             masked_channel = channel.copy()
#             masked_channel[~mask] = 0  # Set pixels outside the mask to zero (black)

#             # Apply bounding box to crop the channel
#             cropped_channel = masked_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]]

#             # Print the shapes for debugging
#             # print("self.output_size:", self.output_size)
#             # print("cropped_channel.shape:", cropped_channel.shape)

#             # Determine breast side (left or right)
#             if bbox[1] == 0:  # Left-aligned breast
#                 # Pad only on the right side
#                 left_pad = 0
#                 right_pad = self.output_size[0] - cropped_channel.shape[1]
#             else:  # Right-aligned breast
#                 # Pad only on the left side
#                 left_pad = self.output_size[0] - cropped_channel.shape[1]
#                 right_pad = 0
            
#              # Crop the cropped channel if it's larger than the output size
#             if cropped_channel.shape[0] > self.output_size[1]:
#                 cropped_channel = cropped_channel[:self.output_size[1], :]
#             if cropped_channel.shape[1] > self.output_size[0]:
#                 cropped_channel = cropped_channel[:, :self.output_size[0]]

#             # Pad the cropped channel accordingly
#             top_pad = 0
#             bottom_pad = self.output_size[1] - cropped_channel.shape[0] - top_pad
#             padded_channel = cv2.copyMakeBorder(cropped_channel, top_pad, bottom_pad, left_pad, right_pad,
#                                                  cv2.BORDER_CONSTANT, value=0)
#             masked_cropped_channels.append(padded_channel)

#         # Merge the cropped channels back into a three-channel image
#         masked_cropped_image = cv2.merge(masked_cropped_channels)

#         return masked_cropped_image

    
# class RandomGaussianNoise(object):
#     def __init__(self, mean=0, std=0.1):
#         self.mean = mean
#         self.std = std

#     def __call__(self, image):
#         noise = torch.randn_like(image) * self.std + self.mean
#         return image + noise

class RandomGaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.asarray(image)
        noise = np.random.normal(self.mean, self.std, image.shape)
        return image + noise.astype(np.uint8)

class RandomSaltPepperNoise(object):
    def __init__(self, salt_prob=0.1, pepper_prob=0.1):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, image):
        # Convert to numpy array
        image_np = np.array(image)

        # Add salt noise
        if random.random() < self.salt_prob:
            num_salt = np.ceil(self.salt_prob * image_np.size)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
            image_np[coords[0], coords[1], :] = 1

        # Add pepper noise
        if random.random() < self.pepper_prob:
            num_pepper = np.ceil(self.pepper_prob * image_np.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
            image_np[coords[0], coords[1], :] = 0

        # Convert back to PIL image
        return Image.fromarray(image_np)
    
class HorizontalFlip(object):
    def __call__(self, img):
        return torch.flip(img, dims=[-1])
    
class VerticalFlip(object):
    def __call__(self, img):
        return torch.flip(img, dims=[-2])

class DataTransformer:
    def __init__(self, **kwargs): #cfg_data = None, set_type='train', transformer=None
        height = kwargs['height']
        width = kwargs['width']
        set_type = kwargs['set_type']
        mean = kwargs['mean']
        std = kwargs['std']

        if set_type == 'train':
            
            transformer = transforms.Compose([
                                # MaskCropping(output_size=(width, height)),
                                # transforms.RandomRotation(20),  # Random rotation with a maximum of 20 degrees
                                # transforms.RandomAffine(0, translate=(0.125, 0.125)),  # Random translation
                                # transforms.RandomHorizontalFlip(),  # Random horizontal flip
                                # transforms.RandomVerticalFlip(),  # Random vertical flip
                                # transforms.RandomAffine(0, scale=(0.8, 1.2)),  # Random zoom
                                # transforms.RandomAffine((-22.5, 22.5), shear=(-0.2, 0.2), scale=(0.8, 1.2), translate=(0.125, 0.125)),
                                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter       
                                # RandomGaussianNoise(mean=0, std=0.1),  # Custom transformation for adding random Gaussian noise
                                # RandomSaltPepperNoise(),  # Custom transformation for adding random salt-pepper noise                         
                                MakeRGB(),
                                transforms.ToTensor()
                                ])
        
        elif set_type == 'pseudo_train':
            transformer = transforms.Compose([
                                MakeRGB(),
                                transforms.ToTensor(),
                                ])
            
        elif set_type == 'train_hflip':
            
            transformer = transforms.Compose([
                                MakeRGB(),
                                transforms.ToTensor(),
                                HorizontalFlip()
                                ])
            
        elif set_type == 'train_vflip':
            
            transformer = transforms.Compose([
                                MakeRGB(),
                                transforms.ToTensor(),
                                VerticalFlip()
                                ])
        
        elif set_type == 'valid' or set_type == 'test':

            transformer = transforms.Compose([
                                # MaskCropping(output_size=(width,height)),
                                MakeRGB(),
                                transforms.ToTensor(),
                                ])

        self.transformer = transformer