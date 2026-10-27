"""
Hand Tracking Module using MediaPipe and OpenCV.
Detects hands, extracts 21 landmark positions, calculates finger states and distances.
"""

import math
import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.results = None
        self.lm_list = []

    def find_hands(self, img, draw=True):
        """Processes the frame to detect hands and draw landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return img

    def find_positions(self, img, hand_no=0, draw=True):
        """Extracts landmark coordinate positions in pixel units."""
        self.lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                h, w, _ = img.shape
                for idx, lm in enumerate(my_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([idx, cx, cy])
                    if draw and idx in self.tip_ids:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        """
        Returns a list of 5 binary values [thumb, index, middle, ring, pinky]
        indicating whether each finger is open (1) or closed (0).
        """
        fingers = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        # Thumb: Compare x-coordinate to landmark 3 (depending on hand orientation)
        # For general usage, check distance between thumb tip (4) and pinky MCP (17)
        # compared to thumb IP (3) and pinky MCP (17).
        thumb_tip_dist = math.hypot(
            self.lm_list[4][1] - self.lm_list[17][1],
            self.lm_list[4][2] - self.lm_list[17][2],
        )
        thumb_ip_dist = math.hypot(
            self.lm_list[3][1] - self.lm_list[17][1],
            self.lm_list[3][2] - self.lm_list[17][2],
        )
        if thumb_tip_dist > thumb_ip_dist:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers: Check if tip y is above pip y (landmark y is inverted: top is 0)
        for idx in range(1, 5):
            tip_idx = self.tip_ids[idx]
            pip_idx = tip_idx - 2
            if self.lm_list[tip_idx][2] < self.lm_list[pip_idx][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img=None, draw=True):
        """Calculates Euclidean distance between two landmarks p1 and p2."""
        if len(self.lm_list) < 21:
            return 0, [0, 0, 0, 0], img

        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

        return length, [x1, y1, x2, y2, cx, cy], img
