import os
import torch
import math
import torchvision
from torch.utils.data import Dataset



def get_file_names(path):
    with open(path) as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    return files

def get_labels(path_to_classes):
    classes ={}
    with open(path_to_classes) as f:
        c = f.readlines()
    for x in c:
        classes[x.strip().split(" ")[1]] =  int(x.strip().split(" ")[0])
    return classes



class RGBFrames(Dataset):

    def __init__(self, text_file, label_path, root_dir ,mode=None, transform = None):
        self.segment_count = 4
        self.root_dir = root_dir
        self.video_paths = get_file_names(text_file)
        self.transform = transform
        self.labels = get_labels(label_path)
        self.mode = mode

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_path = os.path.join(self.root_dir, self.video_paths[idx]+".avi")
        video_category_name = self.video_paths[idx].split("/")[0]
        video = torchvision.io.read_video(video_path, pts_unit='sec')[0]
        total_frames = video.shape[0] # get total frames
        segment_size = int(math.floor(total_frames / self.segment_count)) # divide frames into 4 equal segments
        if self.mode == "test": # if testing then take center frame
            indices = torch.repeat_interleave(torch.Tensor([int(segment_size/2)]), repeats=self.segment_count).int()
        else: # if training take a random frame
            indices = torch.randint(0,segment_size, (self.segment_count,) )
        offset = torch.arange(self.segment_count)*segment_size
        frame_indices = indices + offset # find the frame indices for the particular sample
        snippets = (video[frame_indices].float()/255).permute(0,3,1,2) # extract frames from video using indices
        if self.transform: # apply transform
            snippets = self.transform(snippets)
        return snippets, self.labels[video_category_name], idx


class OpticalFlowStack(Dataset):

    def __init__(self, text_file, label_path, root_dir ,mode=None, transform = None):
        self.stack_size = 5
        self.segment_count = 4
        self.root_dir = root_dir
        self.video_paths = get_file_names(text_file)
        self.transform = transform
        self.labels = get_labels(label_path)
        self.mode = mode

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_category_name = self.video_paths[idx].split("/")[0]
        optical_flow_names = sorted(os.listdir(os.path.join(self.root_dir, self.video_paths[idx]))) # get file names from training.txt
        total_flows = int(len(optical_flow_names)/2) # total flows in a sample
        segment_size = int(math.floor(total_flows / self.segment_count)) # create 4 segments
        if self.mode=="test": # if test take center of segment
            indices = torch.repeat_interleave(torch.Tensor([int(segment_size / 2)]), repeats=self.segment_count).int()
        else: # if train take random
            indices = torch.randint(0,segment_size-self.stack_size, (self.segment_count,) )
        # frame index calculation for flow x and flow y for each segment
        offset = torch.arange(self.segment_count) * segment_size
        frame_start_indices_x = indices + offset
        frame_start_indices_y = frame_start_indices_x + total_flows
        frame_start_indices = torch.vstack([frame_start_indices_x,frame_start_indices_y]).transpose(0,1)
        frame_end_indices = frame_start_indices + self.stack_size
        frame_indices  = torch.stack([frame_start_indices,frame_end_indices],2).flatten(1)
        stack_names = [optical_flow_names[s[0]:s[1]] + optical_flow_names[s[2]:s[3]] for s in frame_indices]
        stack_names = [[os.path.join(self.root_dir, self.video_paths[idx], frame) for frame in stack] for stack in stack_names]
        # extract optical flow for selected indices
        optical_flows = torch.stack([torch.stack([torchvision.io.read_image(frame) for frame in stack]).squeeze(1) for stack in stack_names]).float()/255.0
        if self.transform: # transform the flows
            optical_flows = self.transform(optical_flows)
        return optical_flows,self.labels[video_category_name], idx



