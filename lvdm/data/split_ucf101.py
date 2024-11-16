# Split the UCF-101 official dataset to train and test splits
# The output data formate:
# UCF-101/
# ├── train/
# │   ├── ApplyEyeMakeup/
# │   │   ├── v_ApplyEyeMakeup_g08_c01.avi
# │   │   ├── v_ApplyEyeMakeup_g08_c02.avi
# │   │   ├── ...
# │   ├── ApplyLipstick/
# │   │   ├── v_ApplyLipstick_g01_c01.avi
# ├── test/
# │   ├── ApplyEyeMakeup/
# │   │   ├── v_ApplyEyeMakeup_g01_c01.avi
# │   │   ├── v_ApplyEyeMakeup_g01_c02.avi
# │   │   ├── ...
# │   ├── ApplyLipstick/
# │   │   ├── v_ApplyLipstick_g01_c01.avi
# ├── ucfTrainTestlist/
# │   ├── classInd.txt
# │   ├── testlist01.txt
# │   ├── trainlist01.txt
# │   ├── ...

input_dir = "temp/UCF-101" # the root directory of the UCF-101 dataset
input_annotation = "temp/ucfTrainTestlist" # the annotation file of the UCF-101 dataset
output_dir_tmp = f"{input_dir}_split" # the temporary directory to store the split dataset

remove_original_dir = False # The output directory will be created in the same directory as the input directory

import os
import random
import shutil

split_idx = 1 # the split index
# read the annotation file

# make the train and test directories
os.makedirs(os.path.join(output_dir_tmp, f"train"), exist_ok=True)
os.makedirs(os.path.join(output_dir_tmp, f"test"), exist_ok=True)

def extract_split(subset="train"):
    if subset not in ["train", "test"]:
        raise ValueError("subset must be either 'train' or 'test'")

    with open(os.path.join(input_annotation, f"{subset}list0{split_idx}.txt")) as f:
        train_list = f.readlines()
        train_list = [x.strip() for x in train_list]
        for item in train_list:
            if subset == "test":
                class_name = item.split("/")[0]
                video_name = item.split("/")[1]
            elif subset == "train":
                class_name = item.split("/")[0]
                video_name = item.split(" ")[0].split("/")[1]
            video_path = os.path.join(input_dir, class_name, video_name)
            print(f"input_dir: {input_dir}, class_name: {class_name}, video_name: {video_name}")
            
            class_dir = os.path.join(output_dir_tmp, f"{subset}", class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(video_path, class_dir)
            print(f"Copy {video_path} to {class_dir}")

# split the dataset into the output directory
extract_split(subset="train")
extract_split(subset="test")

# copy the annotation files to the output directory
shutil.copytree(input_annotation, os.path.join(output_dir_tmp, "ucfTrainTestlist"))

if remove_original_dir:
    shutil.rmtree(input_dir)
    shutil.move(output_dir_tmp, input_dir)
