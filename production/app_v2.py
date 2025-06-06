import sys
import cv2
import numpy as np
import torch
import requests
import json
import mediapipe as mp
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QRadioButton, QButtonGroup)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui      import QKeySequence
import os

API_URL = "http://localhost:8000/detect"


def read_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def get_response(payload):
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            pred = response.json().get("prediction", 'Error')
        else:
            print(response.text)
            pred = 'Error'
    except Exception:
        pred = 'Error'
    return pred


def make_segments(buffer, seq_len=16, overlap=0.5):
    """
    Given `buffer` = list of frames, returns a list of windows each of length `seq_len`
    with 50% overlap (i.e. stride = seq_len * (1 - overlap)).  If buffer has fewer
    than seq_len frames it pads with zero-frames; if the last window is partial it
    pads it as well.
    """
    stride = int(seq_len * (1 - overlap))
    N = len(buffer)
    if N == 0:
        return []
    # how many windows to produce
    if N < seq_len:
        num_windows = 1
    else:
        num_windows = ((N - seq_len) + stride - 1) // stride + 1

    # prepare a zero-frame of the same shape as buffer[0]
    frame_len = len(buffer[0])
    zero_frame = [0.0] * frame_len

    segments = []
    for w in range(num_windows):
        start = w * stride
        window = buffer[start : start + seq_len]
        # pad if too short
        if len(window) < seq_len:
            window = window + [zero_frame] * (seq_len - len(window))
        segments.append(window)

    return segments


def draw_keypoints(image, hand_result, pose_result):
    image_height, image_width = image.shape[:2]
    # Drawing style
    kp_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    line_color = (0, 0, 0)

    # Draw pose
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark
        points = {
            'left': [(lm[15], lm[13]), (lm[13], lm[11])],
            'right': [(lm[16], lm[14]), (lm[14], lm[12])]
        }
        for side in points.values():
            for p1, p2 in side:
                x1, y1 = int(p1.x * image_width), int(p1.y * image_height)
                x2, y2 = int(p2.x * image_width), int(p2.y * image_height)
                cv2.line(image, (x1, y1), (x2, y2), line_color, 4)
                for x, y in [(x1, y1), (x2, y2)]:
                    cv2.circle(image, (x, y), 6, outline_color, thickness=2)
                    cv2.circle(image, (x, y), 4, kp_color, thickness=-1)

    # Draw & connect both hands
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # connect landmarks by drawing the standard hand connections
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                p1 = hand_landmarks.landmark[start_idx]
                p2 = hand_landmarks.landmark[end_idx]
                x1, y1 = int(p1.x * image_width), int(p1.y * image_height)
                x2, y2 = int(p2.x * image_width), int(p2.y * image_height)
                cv2.line(image, (x1, y1), (x2, y2), line_color, 4)
            # draw keypoints
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * image_width), int(lm.y * image_height)
                cv2.circle(image, (cx, cy), 6, outline_color, thickness=2)
                cv2.circle(image, (cx, cy), 4, kp_color, thickness=-1)


class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.space_sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_sc.activated.connect(self.toggle)
        self.setWindowTitle("ASL DETECTION")
        self.setStyleSheet("background-color: white;")
        self.setFocusPolicy(Qt.StrongFocus)

        self.class_mapper = read_json("class_mapper.json")
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Mediapipe
        self.mp_hands = mp.solutions.hands.Hands(False, max_num_hands=2)
        self.mp_pose = mp.solutions.pose.Pose(False)

        self.running = False
        self.keypoint_buffer = []
        self.video_writer = None

        # Layout
        self.layout = QVBoxLayout(self)

        # Header
        header = QLabel("ASL DETECTION")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header)

        # Info block
        info_layout = QVBoxLayout()
        self.info_box = QLabel("Info:\nReady")
        self.info_box.setStyleSheet("background-color: lightgray; font-weight: bold; padding: 4px")
        info_layout.addWidget(self.info_box)

        # Toggle buttons
        self.kp_yes = QRadioButton("yes")
        self.kp_no = QRadioButton("no")
        self.kp_yes.setChecked(True)
        self.kp_group = QButtonGroup(self)
        self.kp_group.addButton(self.kp_yes)
        self.kp_group.addButton(self.kp_no)

        self.save_yes = QRadioButton("yes")
        self.save_no = QRadioButton("no")
        self.save_no.setChecked(True)
        self.save_group = QButtonGroup(self)
        self.save_group.addButton(self.save_yes)
        self.save_group.addButton(self.save_no)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Show keypoints:"))
        row1.addWidget(self.kp_yes)
        row1.addWidget(self.kp_no)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Save video:"))
        row2.addWidget(self.save_yes)
        row2.addWidget(self.save_no)
        info_layout.addLayout(row1)
        info_layout.addLayout(row2)
        self.layout.addLayout(info_layout)

        # Image label
        self.image_label = QLabel("IMAGE")
        self.image_label.setFixedHeight(350)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: lightgray; font-weight: bold")
        self.layout.addWidget(self.image_label)

        # Prediction
        self.prediction_label = QLabel("Prediction:")
        self.prediction_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.layout.addWidget(self.prediction_label)

        # Start button
        self.toggle_button = QPushButton("Start")
        self.toggle_button.setStyleSheet("background-color: green; color: white")
        self.toggle_button.clicked.connect(self.toggle)
        self.layout.addWidget(self.toggle_button)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle()

    def toggle(self):
        if not self.running:
            # start
            self.running = True
            self.keypoint_buffer = []
            # initialize video writer if needed
            if self.save_yes.isChecked():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join('videos', f'saved_asl_{timestamp}.avi')
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = 20.0
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                self.info_box.setText(f"Info:\nRecording video to {filename}")
            else:
                self.video_writer = None
                self.info_box.setText("Info:\nRecording (not saving)")

            self.toggle_button.setText("Stop")
            self.toggle_button.setStyleSheet("background-color: red; color: white")
            self.timer.start(50)
        else:
            # stop
            self.running = False
            self.toggle_button.setText("Start")
            self.toggle_button.setStyleSheet("background-color: green; color: white")
            self.timer.stop()
            # finalize video writer
            if self.video_writer:
                self.video_writer.release()
                self.info_box.setText(self.info_box.text() + "\nVideo saved.")
            # run model
            self.send_to_model()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = self.mp_hands.process(frame_rgb)
        pose_result = self.mp_pose.process(frame_rgb)

        if self.kp_yes.isChecked():
            draw_keypoints(frame_rgb, hand_result, pose_result)

        # save frame if recording
        if self.video_writer:
            # write original BGR frame to file
            self.video_writer.write(frame)

        keypoints = self.extract_keypoints(hand_result, pose_result)
        self.keypoint_buffer.append(keypoints)

        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    def extract_keypoints(self, hand_result, pose_result):
        hand_template = np.zeros((21* (len(hand_result.multi_hand_landmarks or [None])), 3), dtype=np.float32)
        # ensure space for two hands
        hand_template = np.zeros((21*2, 3), dtype=np.float32)
        pose_template = np.zeros((6, 3), dtype=np.float32)

        if hand_result.multi_hand_landmarks:
            for h_idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                if h_idx >= 2:
                    break
                for j, lm in enumerate(hand_landmarks.landmark):
                    # print(h_idx, h_idx*21, h_idx*21+j)
                    hand_template[h_idx*21 + j] = [lm.x, lm.y, lm.z]

        if pose_result.pose_landmarks:
            pose_indices = [11, 12, 13, 14, 15, 16]
            for i, idx in enumerate(pose_indices):
                lm = pose_result.pose_landmarks.landmark[idx]
                pose_template[i] = [lm.x, lm.y, lm.z]

        return np.concatenate((hand_template, pose_template), axis=0).flatten().tolist()

    def send_to_model(self):
        self.info_box.setText("Info:\nProcessing prediction...")
        sequence_len = 16
        stride = 8
        # segments = [self.keypoint_buffer[i:i + sequence_len]
        #             for i in range(0, len(self.keypoint_buffer) - sequence_len + 1, stride)]
        segments = make_segments(self.keypoint_buffer)
        # print(len(self.keypoint_buffer))
        # print()
        # print([len(s) for s in segments])
        payload = {"data": segments, "score": "hard"}
        prediction = get_response(payload)
        # prediction = 0
        prediction = self.class_mapper.get(str(prediction), "Unknown")
        self.prediction_label.setText(f"Prediction: {prediction}")
        self.info_box.setText(self.info_box.text() + f"\nPrediction: {prediction}")

    def closeEvent(self, event):
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        self.timer.stop()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec_())
