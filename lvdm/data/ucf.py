import os
import re
import glob
import random
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

def split_by_captical(s):
    s_list = re.sub( r"([A-Z])", r" \1", s).split()
    string = ""
    for s in s_list:
        string += s + " "
    return string.rstrip(" ").lower()

def sample_strided_frames(vid_len, frame_stride, target_vid_len):
    frame_indices = list(range(0, vid_len, frame_stride))
    if len(frame_indices) < target_vid_len:
        frame_stride = vid_len // target_vid_len # recalculate a max fs
        assert(frame_stride != 0)
        frame_indices = list(range(0, vid_len, frame_stride))
    return frame_indices, frame_stride

def class_name_to_idx(annotation_dir):
    """
    return class indices from 0 ~ num_classes-1
    """
    fpath = os.path.join(annotation_dir, "classInd.txt")
    with open(fpath, "r") as f:
        data = f.readlines()
        class_to_idx = {x.strip().split(" ")[1].lower():int(x.strip().split(" ")[0]) - 1 for x in data}
    return class_to_idx

def class_idx_to_caption(caption_path):
    """
    return class captions
    """
    with open(caption_path, "r") as f:
        data = f.readlines()
        idx_to_cap = {i: line.strip() for i, line in enumerate(data)}
    return idx_to_cap

class UCF101(Dataset):
    """
    UCF101 Dataset. Assumes data is structured as follows.

        UCF101/ (data_root)
            train/
                classname
                    xxx.avi
                    xxx.avi
            test/
                classname
                    xxx.avi
                    xxx.avi
            ucfTrainTestlist/
    """
    def __init__(self,
                 data_root,
                 resolution,
                 video_length,
                 subset_split,
                 frame_stride,
                 annotation_dir=None,
                 caption_file=None,
                 ):
        self.data_root = data_root
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.annotation_dir = annotation_dir if annotation_dir is not None else os.path.join(data_root, "ucfTrainTestlist")
        self.caption_file = caption_file

        assert(self.subset_split in ['train', 'test', 'all'])
        self.exts = ['avi', 'mp4', 'webm']
        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        assert(isinstance(self.resolution, list) and len(self.resolution) == 2)

        self._make_dataset()
    
    def _make_dataset(self):
        if self.subset_split == 'all':
            data_folder = self.data_root
        else:
            data_folder = os.path.join(self.data_root, self.subset_split)
        video_paths = sum([glob.glob(os.path.join(data_folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])
        # ucf class_to_idx
        class_to_idx = class_name_to_idx(self.annotation_dir)
        idx_to_cap = class_idx_to_caption(self.caption_path) if self.caption_file is not None else None

        self.videos = video_paths
        self.class_to_idx = class_to_idx
        self.idx_to_cap = idx_to_cap
        print(f'Number of videos = {len(self.videos)}')

    def _get_ucf_classinfo(self, videopath):
        video_name = os.path.basename(videopath)
        class_name = video_name.split("_")[1].lower() # v_BoxingSpeedBag_g12_c05 -> boxingspeedbag
        class_index = self.class_to_idx[class_name] # 0-100
        class_caption = self.idx_to_cap[class_index] if self.caption_file is not None else \
            split_by_captical(video_name.split("_")[1]) # v_BoxingSpeedBag_g12_c05 -> boxing speed bag
        return class_name, class_index, class_caption

    def __getitem__(self, index):
        while True:
            video_path = self.videos[index]

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                vid_len = len(video_reader)
                if vid_len < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
        
        # sample strided frames
        all_frames, frame_stride = sample_strided_frames(vid_len, frame_stride, self.video_length)

        # select random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = list(range(rand_idx, rand_idx+self.video_length))
        frames = video_reader.get_batch(frame_indices)
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        
        class_name, class_index, class_caption =  self._get_ucf_classinfo(videopath=video_path)
        data = {'video': frames, 
                'class_name': class_name, 
                'class_index': class_index, 
                'class_caption': class_caption
                }
        return data
    
    def __len__(self):
        return len(self.videos)