import cv2
import numpy as np
from model import TSN
import os
import torch
from dataset import RGBFrames
from torchvision.transforms import Compose, Normalize, Resize, RandomHorizontalFlip, RandomApply, RandomCrop, RandomResizedCrop
from torchvision import transforms

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
        classes[int(x.strip().split(" ")[0])] =  x.strip().split(" ")[1]
    return classes
path_to_validation = "../data/mini_UCF/validation.txt"
path_to_classes ='../data/mini_UCF/classes.txt'
root_rgb = '../data/mini_UCF/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

validation_files = np.array(get_file_names(path_to_validation))
test_videos_idx= np.random.randint(0,len(validation_files),5)
dataset_test = RGBFrames(path_to_validation,path_to_classes,root_rgb,mode="test",transform = transforms.Compose([Resize((224,224)), Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) ]))
labels = get_labels(path_to_classes)
tsn_rgb = torch.load("trained_models/RGB_tsn.pth")['model'].eval()
for idx in test_videos_idx:
    test_video = dataset_test[idx][0].unsqueeze(0).to(device)
    pred = int(torch.argmax(tsn_rgb(test_video), dim=1).data)
    predicted_action = labels[pred]
    original_video = cv2.VideoCapture(os.path.join(root_rgb,validation_files[idx]+'.avi'))
    total_frames = original_video.get(cv2.CAP_PROP_FRAME_COUNT)
    if (original_video.isOpened() == False):
        print("Error opening video  file")
    count = 0
    while (original_video.isOpened()):
        ret, frame = original_video.read()
        if ret == True:
            if(count>total_frames/2):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, predicted_action, (10,50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            count+=1
            cv2.imshow('Frame', frame)
            cv2.waitKey(2)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    original_video.release()
    cv2.destroyAllWindows()





