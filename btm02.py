import http

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import time
import numpy as np

model_path = "C:/Users/Saaim/Desktop/Computer Vision/pose_landmarker_lite.task"


class BattingEngine:
    def __init__(self, model_path):
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)
        self.scores = []

    def technique_score(self, x_head, x_hands, x_foot, img_w):
        xs = np.array([x_head, x_hands, x_foot])
        mean_x = np.mean(xs)
        deviation = np.mean(np.abs(xs - mean_x))
        norm_dev = deviation / img_w
        score = max(0, 100 - norm_dev * 400)
        return int(score)

    def process(self, frame):

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return {"valid": False}

        lm = result.pose_landmarks[0]

        nose = lm[0]
        left_wrist = lm[15]
        right_wrist = lm[16]
        left_toe = lm[31]
        right_toe = lm[32]

        if min(
            nose.visibility,
            left_wrist.visibility,
            right_wrist.visibility,
            left_toe.visibility,
            right_toe.visibility
        ) < 0.2:
            return {"valid": False}

        x_head = nose.x * w
        y_head = nose.y * h

        x_hands = ((left_wrist.x + right_wrist.x) / 2) * w
        y_hands = ((left_wrist.y + right_wrist.y) / 2) * h

        if left_toe.visibility > right_toe.visibility:
            x_foot = left_toe.x * w
            y_foot = left_toe.y * h
            front_label = "L foot"
        else:
            x_foot = right_toe.x * w
            y_foot = right_toe.y * h
            front_label = "R foot"

        score = self.technique_score(
            x_head, x_hands, x_foot, w
        )

        mean_x = int((x_head + x_hands + x_foot) / 3)

        y1 = int(y_head)
        y2 = int(((left_toe.y + right_toe.y) / 2) * h)

        return {
            "valid": True,
            "score": score,

            "head": (int(x_head), int(y_head)),
            "hands": (int(x_hands), int(y_hands)),
            "foot": (int(x_foot), int(y_foot)),

            "line": (mean_x, y1, y2),

            "front": front_label
        }


engine = BattingEngine(model_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ok, frame = cap.read()
    if not ok:
        continue

    output = engine.process(frame)

    if output["valid"]:

        head = output["head"]
        hands = output["hands"]
        foot = output["foot"]

        mx, y1, y2 = output["line"]

        # --- vertical alignment line (used for scoring)
        cv2.line(frame, (mx, y1), (mx, y2), (0, 255, 255), 2)

        # --- lines actually used to check technique
        cv2.line(frame, head, hands, (255, 0, 0), 2)   # head -> hands
        cv2.line(frame, hands, foot, (255, 0, 0), 2)   # hands -> front toe

        # --- key points
        cv2.circle(frame, head, 6, (0, 255, 0), -1)
        cv2.circle(frame, hands, 6, (0, 255, 0), -1)
        cv2.circle(frame, foot, 6, (0, 255, 0), -1)

        score = output["score"]

        color = (0, 255, 0) if score > 60 else (0, 0, 255)

        cv2.putText(
            frame,
            f"Technique Score: {score}/100",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Front foot: {output['front']}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("Batting Technique Master", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
