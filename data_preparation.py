from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import json 


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

class MSASLVideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=32, img_size=224, transforms=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.load_video(video_path)
        if self.transforms:
            frames = self.transforms(frames)

        return frames, label

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


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


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
