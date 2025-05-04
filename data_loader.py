from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import json 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

class MSASLVideoDataset(Dataset):
    def __init__(self, video_paths, labels, n_labels=100, num_frames=32, img_size=224, transforms=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.transforms = transforms
        self.n_labels = n_labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.load_video(video_path)
        if self.transforms:
            frames = self.transforms(frames)

        ohe_label = torch.from_numpy(int_to_ohe(label, self.n_labels))
        return frames, ohe_label

    def load_video(self, path):
        frames = read_frames(path, self.img_size)
        # Uniformly sample self.num_frames frames
        total_frames = frames.shape[0]
        if total_frames >= self.num_frames:
            idxs = np.linspace(0, total_frames-1, self.num_frames).astype(int)
            frames = frames[idxs]
        else:
            # pad by repeating last frame
            pad_len = self.num_frames - total_frames
            pad_frames = np.repeat(frames[-1:], pad_len, axis=0)
            frames = np.concatenate((frames, pad_frames), axis=0)

        # frames = frames.transpose(0, 3, 1, 2)  # (Frames, Channels, Height, Width)
        frames = torch.from_numpy(frames).float() / 255.0  # normalize 0-1
        return frames
    

class MSASLPreProcessedVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transforms=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = np.load(video_path)
        if self.transforms:
            frames = self.transforms(frames)
        
        frames = torch.from_numpy(frames)
        ohe_label = torch.from_numpy(int_to_ohe(label, self.n_labels))
        return frames, ohe_label
    
class MSASLKeypointsDataset(Dataset):
    def __init__(self, kpts_paths, labels, transforms=None):
        self.kpts_paths = kpts_paths
        self.labels = labels

    def __len__(self):
        return len(self.kpts_paths)

    def __getitem__(self, idx):
        kpts_path = self.kpts_paths[idx]
        label = self.labels[idx]
        metadata = load_json(kpts_path)
        hand_kpts = metadata['hand_keypoints']
        pose_kpts = metadata['pose_keypoints']
        all_keypoints = []
        for frame_ind in range(len(hand_kpts)):
            hand_template = np.zeros((21*2,3))
            pose_template = np.zeros((6,3))
            for i,kpt in enumerate(hand_kpts[frame_ind]):
                hand_template[i] = kpt[0]
            for i,kpt in enumerate(pose_kpts[frame_ind]):
                pose_template[i] = kpt
            merge_kpts = np.concat((hand_template, pose_template)).tolist()
            all_keypoints.append(merge_kpts)
        
        all_keypoints = torch.Tensor(all_keypoints)
        ohe_label = torch.from_numpy(int_to_ohe(label, self.n_labels))
        return all_keypoints, ohe_label

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def int_to_ohe(ind, n):
    ohe = np.zero(shape = (n,))
    ohe[ind] = 1
    return ohe
    
def read_frames(path, img_size):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    cap.release()

    frames = np.array(frames)
    return frames

def plot_video_gif(video_tensor, fps=5, label=None):
    """
    Displays a video tensor as an animated GIF inline.
    
    Args:
        video_tensor: A torch tensor or numpy array with shape [T, H, W, C] or [T, C, H, W]
        fps: Frames per second for playback
    """
    fig = plt.figure(figsize=(6, 6))
    img = plt.imshow(video_tensor[0])

    def animate(i):
        img.set_array(video_tensor[i])
        return [img]
    plt.axis('off')
    if label:
        plt.title(label)
    ani = animation.FuncAnimation(fig, animate, frames=len(video_tensor), interval=1000/fps, blit=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())
