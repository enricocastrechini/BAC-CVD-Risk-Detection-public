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
import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def determine_aspect_ratio(width, height):
        return width / height

def decide_crop_orientation(original_aspect_ratio, target_aspect_ratio):
    if original_aspect_ratio > target_aspect_ratio:
        return "horizontal"
    else:
        return "vertical"

def crop_image(image, orientation, target_width, target_height):
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

def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height))

def apply_letterboxing(image, target_width, target_height):
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


def process_image(image_path, output_bb_dir, output_prep_dir, output_masks_dir, output_size):
    # Preprocess the image
    image = cv2.imread(image_path)
    image = np.asarray(image)
    channels = cv2.split(image)
    masked_cropped_channels = []
    cropped_channels = []
    mask_channels = []

    for i, channel in enumerate(channels):
        # Segment the mammogram for each channel separately
        segmenter = BreastSegmenter()
        mask, bbox = segmenter(channel)

        # Convert bool mask to uint8 (0 and 255)
        mask_uint8 = mask.astype(np.uint8) * 255

        masked_channel = channel.copy()
        masked_channel[~mask] = 0  # Set pixels outside the mask to zero (black)

        # Apply bounding box to crop the channel
        cropped_channel = masked_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]] 
        cropped_channel.shape[1] // 2
        cropped_channel.shape[0] // 2

        # Resize the cropped channel to the fixed size
        resized_channel = resize_image(cropped_channel, output_size[0], output_size[1])

        # Integrate the cropping and resizing algorithm
        original_aspect_ratio = determine_aspect_ratio(cropped_channel.shape[1], cropped_channel.shape[0])
        target_aspect_ratio = determine_aspect_ratio(output_size[0], output_size[1])
        crop_orientation = decide_crop_orientation(original_aspect_ratio, target_aspect_ratio)
        cropped_resized_channel = crop_image(resized_channel, crop_orientation, output_size[0], output_size[1])

        # Apply letterboxing if necessary
        final_channel = apply_letterboxing(cropped_resized_channel, output_size[0], output_size[1])

        # Append the processed channel to the list
        masked_cropped_channels.append(final_channel)
        cropped_channels.append(cropped_channel)
        mask_channels.append(mask_uint8)

    # Merge the cropped channels back into a three-channel image
    masked_cropped_image = cv2.merge(masked_cropped_channels)
    cropped_image = cv2.merge(cropped_channels)
    mask_image = cv2.merge(mask_channels)

    # Write the masked and cropped image
    filename = os.path.basename(image_path)
    output_bb_path = os.path.join(output_bb_dir, filename)
    output_prep_path = os.path.join(output_prep_dir, filename)
    output_masks_path = os.path.join(output_masks_dir, filename)

    # cv2.imwrite(output_bb_path, cropped_image)  # Save cropped image
    cv2.imwrite(output_prep_path, masked_cropped_image)  # Save processed image
    # cv2.imwrite(output_masks_path, mask_image)  # Save mask

    return masked_cropped_image, cropped_image, mask_uint8

def process_image_hd(image_path, output_prep_dir, output_size):
    # Preprocess the image
    image = cv2.imread(image_path)
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

        # Print the shapes for debugging
        # print("self.output_size:", output_size)
        # print("cropped_channel.shape:", cropped_channel.shape)

        # Determine breast side (left or right)
        if bbox[1] == 0:  # Left-aligned breast
            # Pad only on the right side
            left_pad = 0
            right_pad = output_size[0] - cropped_channel.shape[1]
        else:  # Right-aligned breast
            # Pad only on the left side
            left_pad = output_size[0] - cropped_channel.shape[1]
            right_pad = 0
        
        # Crop the cropped channel if it's larger than the output size
        if cropped_channel.shape[0] > output_size[1]:
            cropped_channel = cropped_channel[:output_size[1], :]
            # cropped_channel = cv2.resize(cropped_channel, (output_size[0], output_size[1]))

        if cropped_channel.shape[1] > output_size[0]:
            cropped_channel = cropped_channel[:, :output_size[0]]
            # cropped_channel = cv2.resize(cropped_channel, (output_size[0], output_size[1]))


        # Pad the cropped channel accordingly
        top_pad = 0
        bottom_pad = output_size[1] - cropped_channel.shape[0] - top_pad
        padded_channel = cv2.copyMakeBorder(cropped_channel, top_pad, bottom_pad, left_pad, right_pad,
                                                cv2.BORDER_CONSTANT, value=0)
        masked_cropped_channels.append(padded_channel)

    # Merge the cropped channels back into a three-channel image
    masked_cropped_image = cv2.merge(masked_cropped_channels)

    masked_cropped_image = cv2.resize(masked_cropped_image, (masked_cropped_image.shape[1] // 2, masked_cropped_image.shape[0] // 2))


     # Write the masked and cropped image
    filename = os.path.basename(image_path)
    output_prep_path = os.path.join(output_prep_dir, filename)

    cv2.imwrite(output_prep_path, masked_cropped_image)  # Save processed image

    return masked_cropped_image

if __name__ == "__main__":
    # pos_dir = "dataset/exams_with_vascular_calcs"
    # neg_dir = "dataset/cases_not_checked_for_bac"
    neg_dir = os.environ.get('BAC_NEGATIVE_IMAGE_DIR')
    if not neg_dir:
        raise ValueError('Set BAC_NEGATIVE_IMAGE_DIR to an authorized local image directory.')

    output_bb_pos_dir = "dataset/processed_hbb_5000/bb/exams_with_vascular_calcs"
    output_prep_pos_dir = "dataset/processed_hbb_5000/exams_with_vascular_calcs"
    output_masks_pos_dir = "dataset/processed_hbb_5000/masks/exams_with_vascular_calcs"

    output_bb_neg_dir = "dataset/processed_hbb_5000/bb/cases_not_checked_for_bac"
    output_prep_neg_dir = "dataset/processed_hbb_5000"
    output_masks_neg_dir = "dataset/processed_hbb_5000/masks/cases_not_checked_for_bac"

    output_size = (576, 1120)


    # Ensure output directories exist
    for path in [output_bb_pos_dir, output_prep_pos_dir, output_masks_pos_dir,
                 output_bb_neg_dir, output_prep_neg_dir, output_masks_neg_dir]:
        os.makedirs(path, exist_ok=True)

    # # Process positive images
    # all_image_paths_pos = [os.path.join(pos_dir, filename) for filename in os.listdir(pos_dir) if filename.endswith((".png", ".jpg"))]
    # with Pool() as pool:
    #     partial_process_image_pos = partial(process_image, output_bb_dir=output_bb_pos_dir, output_prep_dir=output_prep_pos_dir, output_masks_dir=output_masks_pos_dir, output_size=output_size)
    #     # partial_process_image_pos = partial(process_image_hd, output_prep_dir=output_prep_pos_dir, output_size=output_size)
    #     results_pos = list(tqdm(pool.imap(partial_process_image_pos, all_image_paths_pos), total=len(all_image_paths_pos), desc="Processing positive images"))

    # print("Positive images processed and saved successfully.")

    # Process negative images
    all_image_paths_neg = [os.path.join(neg_dir, filename) for filename in os.listdir(neg_dir) if filename.endswith((".png", ".jpg"))]
    with Pool() as pool:
        partial_process_image_neg = partial(process_image, output_bb_dir=output_bb_neg_dir, output_prep_dir=output_prep_neg_dir, output_masks_dir=output_masks_neg_dir, output_size=output_size)
        # partial_process_image_neg = partial(process_image_hd, output_prep_dir=output_prep_neg_dir, output_size=output_size)
        results_neg = list(tqdm(pool.imap(partial_process_image_neg, all_image_paths_neg), total=len(all_image_paths_neg), desc="Processing negative images"))

    print("Negative images processed and saved successfully.")
