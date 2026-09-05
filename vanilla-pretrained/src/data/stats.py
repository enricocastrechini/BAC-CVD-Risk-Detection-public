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

from breast_segment import BreastSegmenter
import cv2
import numpy as np
import os
from tqdm import tqdm

# pos_dir = "dataset/exams_with_vascular_calcs"
# neg_dir = "dataset/cases_not_checked_for_bac"

# # Initialize variables to accumulate pixel values and counts
# pixel_sum = 0
# pixel_sum_sq = 0
# pixel_count = 0

# def get_mean_std(img):
#         breast_images = []

#         # Apply breast segmentation
#         segmenter = BreastSegmenter()

#         mask, bbox = segmenter(img)

#         breast_region = img[mask]

#         breast_region = np.array(breast_region)

#         mean = breast_region.mean()
#         std = breast_region.std()

#         return mean, std

# # Iterate through each image in both directories
# for directory in [pos_dir, neg_dir]:
#     for filename in os.listdir(directory):
#         if filename.endswith(".png") or filename.endswith(".jpg"):
#             # Preprocess the image
#             image_path = os.path.join(directory, filename)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         

#             # Calculate mean and standard deviation within the breast region
#             mean, std = get_mean_std(image)

#             # Accumulate pixel values, square of pixel values, and counts
#             pixel_sum += mean # Assuming mean is a scalar
#             pixel_sum_sq += (std ** 2)  # Squared standard deviation
#             pixel_count += 1

# # Calculate overall mean and standard deviation
# overall_mean = pixel_sum / pixel_count
# overall_std = np.sqrt(pixel_sum_sq / pixel_count)

# # Print overall mean and standard deviation
# print("Overall Mean:", overall_mean)
# print("Overall Standard Deviation:", overall_std)


        
# def get_mean_std(img):
#         breast_regions = []

#         segmenter = BreastSegmenter()
#         mask, bbox = segmenter(img)
#         breast_region = img[mask]
#         breast_region = np.array(breast_region)

#         breast_regions.append(breast_region)
#         breast_regions = np.array(breast_regions)
#         print(breast_regions.shape)
#         mean = breast_regions.mean(axis=(0,1))
#         std = breast_regions.std(axis=(0,1))

#         return mean, std

# if __name__ == "__main__":
#     pos_dir = "dataset/exams_with_vascular_calcs"
#     neg_dir = "dataset/cases_not_checked_for_bac"

#     # Iterate through each image in both directories
#     for directory in [pos_dir, neg_dir]:
#         for filename in os.listdir(directory):
#             if filename.endswith(".png") or filename.endswith(".jpg"):
#                 # Preprocess the image
#                 image_path = os.path.join(directory, filename)
#                 image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#                 # Calculate mean and standard deviation within the breast region
#                 mean, std = get_mean_std(image)

#                 # Print mean and standard deviation for each image
#     print("Mean:", mean)
#     print("Standard Deviation:", std)




# def get_mean_std(img):

#     segmenter = BreastSegmenter()
#     mask, bbox = segmenter(img)
#     breast_region = img[mask]
#     breast_region = np.array(breast_region)
#     print(breast_region.shape)

#     return breast_region

# if __name__ == "__main__":
#     pos_dir = "dataset/exams_with_vascular_calcs"
#     neg_dir = "dataset/cases_not_checked_for_bac"

#     breast_regions = []

#     # Iterate through each image in both directories
#     for directory in [pos_dir, neg_dir]:
#         for filename in tqdm(os.listdir(directory), desc="Processing images"):
#             if filename.endswith(".png") or filename.endswith(".jpg"):
#                 # Preprocess the image
#                 image_path = os.path.join(directory, filename)
#                 image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#                 # Get breast regions from the image
#                 breast_region = get_mean_std(image)

#                 # Accumulate breast regions from all images
#                 breast_regions.extend(breast_region)

#     # Concatenate all breast regions
#     breast_regions = np.concatenate(breast_regions)

#     # Calculate mean and standard deviation over the entire dataset
#     mean = breast_regions.mean(axis=0)
#     std = breast_regions.std(axis=0)

#     # Print mean and standard deviation over the entire dataset
#     print("Mean:", mean)
#     print("Standard Deviation:", std)


import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def segment_breast(img):
    segmenter = BreastSegmenter()
    mask, bbox = segmenter(img)
    breast_region = img[mask]
    breast_region = np.array(breast_region)
    return breast_region

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    breast_region = segment_breast(image)
    return breast_region

if __name__ == "__main__":
    pos_dir = "dataset/exams_with_vascular_calcs"
    neg_dir = "dataset/cases_not_checked_for_bac"

    all_image_paths = []

    # Gather image paths from both directories
    for directory in [pos_dir, neg_dir]:
        all_image_paths.extend([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith((".png", ".jpg"))])

    # Process images using multiprocessing pool
    with Pool() as pool:
        breast_regions = list(tqdm(pool.imap(process_image, all_image_paths), total=len(all_image_paths), desc="Processing images"))

    # Concatenate all breast regions
    breast_regions = np.concatenate(breast_regions)

    # Calculate mean and standard deviation over the entire dataset
    mean = breast_regions.mean(axis=0)
    std = breast_regions.std(axis=0)

    # Print mean and standard deviation over the entire dataset
    print("Mean:", mean/255)
    print("Standard Deviation:", std/255)

# Mean: 73.88263250637023
# Standard Deviation: 45.87464576075177

# Normalized Mean: 0.28935949767803943
# Normalized Standard Deviation: 0.17990232913372892