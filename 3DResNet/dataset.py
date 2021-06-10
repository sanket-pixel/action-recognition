import os
import torch
import torchvision
from torch.utils.data import Dataset


def get_file_names(path):
    # get training or testing file names from text file
    with open(path) as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    return files

def get_labels(path_to_classes):
    # get label id from class name
    classes ={}
    with open(path_to_classes) as f:
        c = f.readlines()
    for x in c:
        classes[x.strip().split(" ")[1]] =  int(x.strip().split(" ")[0])
    return classes




class RGB3DBlocks(Dataset):

    def __init__(self, num_frames, path_text_file, label_path, root_dir, mode = None, transform = None):
        self.num_frames = num_frames
        self.root_dir = root_dir
        self.video_paths = get_file_names(path_text_file)
        self.transform = transform
        self.labels = get_labels(label_path)
        self.mode = mode

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_category_name = self.video_paths[idx].split("/")[0] # get video category name
        video_path = os.path.join(self.root_dir, self.video_paths[idx] + ".avi") # get video path
        video = torchvision.io.read_video(video_path, pts_unit='sec')[0] # read video
        total_frames = len(video) # total frames
        frame_start_idx = torch.randint(0, total_frames-self.num_frames, (1,)) # get random start point of stack
        frame_end_idx = frame_start_idx + self.num_frames # get 16 frames around the selected index
        frames = (video[frame_start_idx:frame_end_idx].float()/255).permute(0,3,1,2) # extract frames and change shape
        if self.transform: # transform
            frames = self.transform(frames)
        frames = frames.permute(1,0,2,3) # change shape to make it a tensor
        return frames, self.labels[video_category_name]





