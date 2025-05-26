import sys
import cv2
import numpy as np
import torch
import requests
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp

API_URL = "http://localhost:8000/detect"

class KeypointApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Keypoint Detection")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.info_label = QLabel('info: ', self)
        self.prediction_label = QLabel("Prediction: ", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)

        self.toggle_button = QPushButton("Start", self)
        self.toggle_button.setStyleSheet("background-color: green; color: white")
        self.toggle_button.clicked.connect(self.toggle_capture)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.prediction_label)
        self.layout.addWidget(self.toggle_button)
        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)

        self.cap = cv2.VideoCapture(0)

        self.running = False
        self.frame_buffer = []
        self.keypoint_buffer = []

        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_capture()

    def toggle_capture(self):
        if not self.running:
            self.keypoint_buffer = []
            self.info_label.setText("info: Capturing keypoints...")
            self.toggle_button.setText("Stop")
            self.toggle_button.setStyleSheet("background-color: red; color: white")
            self.running = True
            self.timer.start(50)  # 20 FPS
        else:
            self.info_label.setText("info: Stopped capturing keypoints...")
            self.running = False
            self.timer.stop()
            self.toggle_button.setText("Start")
            self.toggle_button.setStyleSheet("background-color: green; color: white")
            self.process_video()

    def extract_keypoints(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_result = self.mp_hands.process(image_rgb)
        pose_result = self.mp_pose.process(image_rgb)

        hand_template = np.zeros((42, 3), dtype=np.float32)  # 21*2
        pose_template = np.zeros((6, 3), dtype=np.float32)

        if hand_result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                if i >= 2: break
                for j, lm in enumerate(hand_landmarks.landmark):
                    if j < 21:
                        idx = j + i * 21
                        hand_template[idx] = [lm.x, lm.y, lm.z]  

        if pose_result.pose_landmarks:
            pose_indices = [11, 12, 13, 14, 15, 16]
            for i, j in enumerate(pose_indices):
                lm = pose_result.pose_landmarks.landmark[j]
                pose_template[i] = [lm.x, lm.y, lm.z]  

        merged = np.concatenate((hand_template, pose_template), axis=0).flatten()
        return merged.tolist()

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame = frame.copy()
        keypoints = self.extract_keypoints(frame)
        self.keypoint_buffer.append(keypoints)

        # Display frame
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def process_video(self):
        self.info_label.setText("info: Processing the captured frames...")
        sequence_len = 16
        stride = sequence_len // 2  # 50% overlap
        segments = []
        for i in range(0, len(self.keypoint_buffer) - sequence_len + 1, stride):
            segment = self.keypoint_buffer[i:i + sequence_len]
            segments.append(segment)

        predictions = []
        
        self.info_label.setText("info: Detecting the signs...")
        for segment in segments:
            payload = {"data": segment}
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    pred = response.json()["prediction"]
                    predictions.append(pred)
                else:
                    predictions.append("error")
            except Exception as e:
                predictions.append("error")

        if predictions:
            self.prediction_label.setText(f"Predictions: {predictions}")
        else:
            self.prediction_label.setText("No valid predictions.")
        self.info_label.setText("info: ")

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KeypointApp()
    window.show()
    sys.exit(app.exec_())
