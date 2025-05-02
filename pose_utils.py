import mediapipe as mp
import torch 
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# face_contour_style = mp_drawing_styles.get_default_face_mesh_contours_style()
# face_tesselation_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
hand_landmark_style = mp_drawing_styles.get_default_hand_landmarks_style()
hand_connection_style = mp_drawing_styles.get_default_hand_connections_style()
pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()

# Reduce the marker size and line thickness
for k in hand_landmark_style:
    hand_landmark_style[k].circle_radius = 1
    hand_landmark_style[k].thickness = 1

for k in hand_connection_style:
    hand_connection_style[k].thickness = 1


UPPER_BODY_LANDMARKS = [
    mp_holistic.PoseLandmark.LEFT_SHOULDER,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW,
    mp_holistic.PoseLandmark.RIGHT_ELBOW,
    mp_holistic.PoseLandmark.LEFT_WRIST,
    mp_holistic.PoseLandmark.RIGHT_WRIST,
]

# Define which connections to draw (upper body only)
POSE_CONNECTIONS_UPPER_BODY = [
    (mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.LEFT_ELBOW),
    (mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_SHOULDER),
    (mp_holistic.PoseLandmark.RIGHT_WRIST, mp_holistic.PoseLandmark.RIGHT_ELBOW),
    (mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_SHOULDER),
    (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER),
]


def get_keypoints(df, ind):
  with mp_holistic.Holistic(
      static_image_mode=False,
      model_complexity=2,
      smooth_landmarks=True,
      enable_segmentation=False,
      refine_face_landmarks=False, 
      min_detection_confidence=0.4,
      min_tracking_confidence=0.6,
  ) as holistic:
      vid = df[ind][0]
      all_hand_keypoints = []
      all_pose_keypoints = []
      all_images = []

      for v in vid:
          v_np = (v * 255).type(torch.uint8).clip(0, 255).numpy()
          image = v_np.copy()
          results = holistic.process(image)

          keypoints_frame = []
          pose_frame = []

          image = np.zeros(shape=(224,224,3), dtype=np.uint8)
          if results.left_hand_landmarks:
              mp_drawing.draw_landmarks(
                  image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                  landmark_drawing_spec=hand_landmark_style, connection_drawing_spec=hand_connection_style)
              for landmark in results.left_hand_landmarks.landmark:
                  x, y, z = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]), landmark.z
                  keypoints_frame.append(((x, y, z), 'left'))

          if results.right_hand_landmarks:
              mp_drawing.draw_landmarks(
                  image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                  landmark_drawing_spec=hand_landmark_style, connection_drawing_spec=hand_connection_style)
              for landmark in results.right_hand_landmarks.landmark:
                  x, y, z = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]), landmark.z
                  keypoints_frame.append(((x, y, z), 'right'))

          # Get upper-body pose keypoints
          if results.pose_landmarks:
              pose_landmarks = results.pose_landmarks.landmark
              h, w, _ = image.shape

              # Extract & collect upper-body keypoints
              for landmark_id in UPPER_BODY_LANDMARKS:
                  lm = pose_landmarks[landmark_id]
                  x, y, z = int(lm.x * w), int(lm.y * h), lm.z
                  pose_frame.append((x, y, z))

              # Draw selected upper-body pose connections
              for connection in POSE_CONNECTIONS_UPPER_BODY:
                  start_idx, end_idx = connection
                  start = pose_landmarks[start_idx]
                  end = pose_landmarks[end_idx]

                  x0, y0 = int(start.x * w), int(start.y * h)
                  x1, y1 = int(end.x * w), int(end.y * h)
                  cv2.line(image, (x0, y0), (x1, y1), (255, 255, 0), 2)

          all_hand_keypoints.append(keypoints_frame)
          all_pose_keypoints.append(pose_frame)
          all_images.append(image)

      all_images = np.array(all_images)
      return all_hand_keypoints, all_pose_keypoints, all_images


