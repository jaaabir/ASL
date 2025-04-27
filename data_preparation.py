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
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)

        cap.release()

        frames = np.array(frames)

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

        frames = frames.transpose(0, 3, 1, 2)  # (Frames, Channels, Height, Width)
        frames = torch.from_numpy(frames).float() / 255.0  # normalize 0-1
        return frames
