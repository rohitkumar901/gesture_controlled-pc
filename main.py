"""
AI Hand Gesture Controller for PC (Single File - All in One)
Controls mouse cursor, clicks, scrolling, and system volume using Webcam & Hand Gestures.

No extra custom files needed!
"""

import math
import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# PyAutoGUI settings for high responsiveness
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# Volume control support via Pycaw (Windows)
HAS_PYCAW = False
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = interface.QueryInterface(IAudioEndpointVolume)
    vol_range = volume_ctrl.GetVolumeRange()
    min_vol, max_vol = vol_range[0], vol_range[1]
    HAS_PYCAW = True
except Exception:
    HAS_PYCAW = False


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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return img

    def find_positions(self, img, hand_no=0, draw=True):
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
        fingers = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        # Thumb detection
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

        # 4 Fingers detection
        for idx in range(1, 5):
            tip_idx = self.tip_ids[idx]
            pip_idx = tip_idx - 2
            if self.lm_list[tip_idx][2] < self.lm_list[pip_idx][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img=None, draw=True):
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


def main():
    cam_w, cam_h = 640, 480
    screen_w, screen_h = pyautogui.size()

    frame_reduction = 100
    smoothening = 5
    prev_x, prev_y = 0, 0
    cur_x, cur_y = 0, 0

    last_left_click = 0
    last_right_click = 0
    click_cooldown = 0.35
    is_dragging = False
    prev_scroll_y = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, cam_w)
    cap.set(4, cam_h)

    detector = HandDetector(max_hands=1, detection_con=0.75, track_con=0.75)
    prev_time = 0

    print("=" * 60)
    print(" AI Hand Gesture Controller Started!")
    print(" Camera window me 'q' dabane par program stop hoga.")
    print("=" * 60)

    while True:
        success, img = cap.read()
        if not success:
            print("Webcam connect nahi ho paya!")
            break

        img = cv2.flip(img, 1)

        cv2.rectangle(
            img,
            (frame_reduction, frame_reduction),
            (cam_w - frame_reduction, cam_h - frame_reduction),
            (255, 180, 0),
            2,
        )

        img = detector.find_hands(img, draw=True)
        lm_list = detector.find_positions(img, draw=False)

        current_mode = "Standby (Show Hand)"
        current_time = time.time()

        if len(lm_list) != 0:
            fingers = detector.fingers_up()
            thumb, index, middle, ring, pinky = fingers

            idx_x, idx_y = lm_list[8][1], lm_list[8][2]
            mid_x, mid_y = lm_list[12][1], lm_list[12][2]

            # 1. VOLUME CONTROL
            if pinky == 1 and ring == 0 and middle == 0 and HAS_PYCAW:
                current_mode = "Volume Control"
                vol_dist, _, _ = detector.find_distance(4, 8, img)
                vol = np.interp(vol_dist, [25, 140], [min_vol, max_vol])
                vol_bar = np.interp(vol_dist, [25, 140], [400, 150])
                vol_percent = np.interp(vol_dist, [25, 140], [0, 100])
                volume_ctrl.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"{int(vol_percent)}%", (45, 435), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 255, 0), 2)

            # 2. SCROLL MODE
            elif index == 1 and middle == 1 and ring == 0 and pinky == 0:
                current_mode = "Scrolling"
                cv2.circle(img, (idx_x, idx_y), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (mid_x, mid_y), 10, (0, 255, 255), cv2.FILLED)

                if prev_scroll_y != 0:
                    scroll_diff = prev_scroll_y - idx_y
                    if abs(scroll_diff) > 8:
                        scroll_amount = int(scroll_diff * 4)
                        pyautogui.scroll(scroll_amount)
                prev_scroll_y = idx_y

            # 3. DRAG & DROP
            elif sum(fingers) == 0:
                current_mode = "Drag & Drop (Holding)"
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                cv2.putText(img, "HOLDING", (cam_w // 2 - 60, cam_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 4. MOUSE MOVEMENT & CLICKS
            else:
                prev_scroll_y = 0
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

                dist_left_click, _, _ = detector.find_distance(4, 8, img, draw=False)
                dist_right_click, _, _ = detector.find_distance(4, 12, img, draw=False)

                if dist_left_click < 35:
                    current_mode = "Left Click!"
                    cv2.circle(img, (idx_x, idx_y), 14, (0, 255, 0), cv2.FILLED)
                    if current_time - last_left_click > click_cooldown:
                        pyautogui.click(button="left")
                        last_left_click = current_time

                elif dist_right_click < 35:
                    current_mode = "Right Click!"
                    cv2.circle(img, (mid_x, mid_y), 14, (0, 0, 255), cv2.FILLED)
                    if current_time - last_right_click > click_cooldown:
                        pyautogui.click(button="right")
                        last_right_click = current_time

                elif index == 1 and ring == 0 and pinky == 0:
                    current_mode = "Moving Cursor"
                    cv2.circle(img, (idx_x, idx_y), 10, (255, 0, 255), cv2.FILLED)

                    x_target = np.interp(idx_x, (frame_reduction, cam_w - frame_reduction), (0, screen_w))
                    y_target = np.interp(idx_y, (frame_reduction, cam_h - frame_reduction), (0, screen_h))

                    cur_x = prev_x + (x_target - prev_x) / smoothening
                    cur_y = prev_y + (y_target - prev_y) / smoothening

                    pyautogui.moveTo(cur_x, cur_y)
                    prev_x, prev_y = cur_x, cur_y
                else:
                    current_mode = "Standby (Open Palm)"

        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Mode: {current_mode}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 0), 2)

        cv2.imshow("AI Hand Gesture Controller (Press 'q' to Exit)", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if is_dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
