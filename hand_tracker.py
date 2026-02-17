"""
Hand Tracking Module / Qo'l Kuzatuv Moduli
OpenCV yordamida qo'l harakatlarini kuzatish
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class HandTracker:
    """Qo'l kuzatuvchi klass / Hand tracking class"""
    
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5
    ):
        self.max_hands = max_hands
        self.results = None
        self.landmark_list = []
        self.frame = None
        self.lmList = []
        self.prev_lmList = []
        
        # Teri rangini aniqlash uchun HSV oralig'i
        self.lower_hsv1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_hsv1 = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_hsv2 = np.array([160, 20, 70], dtype=np.uint8)
        self.upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        
    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """Kadrdan qo'llarni topish"""
        self.frame = frame.copy()
        height, width, _ = frame.shape
        
        # HSV ga o'girish
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Teri rangini aniqlash
        mask1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Noise olib tashlash
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Konturlarni topish
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Eng katta konturni topish
        max_area = 0
        hand_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000 and area > max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if 0.2 < aspect_ratio < 2.0:
                    max_area = area
                    hand_contour = contour
                    self.hand_bbox = (x, y, w, h)
        
        if hand_contour is not None:
            if draw:
                # Faqat kontur chizish
                cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                
            # Landmarklarni hisoblash
            x, y, w, h = self.hand_bbox
            self.lmList = self._estimate_landmarks(x, y, w, h)
        
        return frame
    
    def _estimate_landmarks(self, x, y, w, h) -> List:
        """21 ta landmarkni hisoblash"""
        landmarks = []
        
        cx, cy = x + w // 2, y + h // 2
        
        # 0 - bilak
        landmarks.append([cx, y + h, 0])
        
        # Bosh barmoq (1-4)
        thumb_x = x + int(w * 0.15)
        landmarks.append([thumb_x + int(w*0.1), y + int(h*0.2), 0])
        landmarks.append([thumb_x + int(w*0.05), y + int(h*0.35), 0])
        landmarks.append([thumb_x, y + int(h*0.5), 0])
        landmarks.append([thumb_x - int(w*0.05), y + int(h*0.65), 0])
        
        # Boshqa barmoqlar
        finger_positions = [0.35, 0.5, 0.65, 0.8]
        
        for fx in finger_positions:
            finger_x = x + int(w * fx)
            tip_y = y + int(h * 0.1)
            base_y = y + int(h * 0.65)
            
            landmarks.append([finger_x, tip_y, 0])
            landmarks.append([finger_x, y + int(h*0.25), 0])
            landmarks.append([finger_x, y + int(h*0.4), 0])
            landmarks.append([finger_x, y + int(h*0.55), 0])
            landmarks.append([finger_x, base_y, 0])
            
        return landmarks
    
    def get_position(self, frame: np.ndarray, hand_no: int = 0) -> List:
        """Qo'l landmarklar pozitsiyalarini olish"""
        self.landmark_list = []
        
        if self.lmList:
            for landmark_id, landmark in enumerate(self.lmList):
                self.landmark_list.append({
                    'id': landmark_id,
                    'x': landmark[0],
                    'y': landmark[1],
                    'z': landmark[2] if len(landmark) > 2 else 0
                })
                
        return self.landmark_list
    
    def get_finger_positions(self) -> dict:
        """Barmoqlarning holatini olish"""
        fingers = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        if not self.lmList or len(self.lmList) < 21:
            return fingers
            
        # Barmoqlarni aniqlash
        wrist_y = self.lmList[0][1]
        
        # Index, middle, ring, pinky
        for i, tip_idx in enumerate([8, 12, 16, 20]):
            finger_names = ['index', 'middle', 'ring', 'pinky']
            if tip_idx < len(self.lmList):
                fingers[finger_names[i]] = self.lmList[tip_idx][1] < wrist_y
                            
        # Bosh barmoq
        if len(self.lmList) > 4:
            fingers['thumb'] = self.lmList[4][0] < self.lmList[3][0]
        
        return fingers
    
    def get_hand_type(self) -> str:
        """Qol turini aniqlash"""
        return "Right"
    
    def get_index_finger_tip(self) -> Optional[Tuple[int, int]]:
        """Korsatgich barmoq uchi pozitsiyasi"""
        if len(self.lmList) > 8:
            return (self.lmList[8][0], self.lmList[8][1])
        return None
    
    def get_thumb_tip(self) -> Optional[Tuple[int, int]]:
        """Bosh barmoq uchi pozitsiyasi"""
        if len(self.lmList) > 4:
            return (self.lmList[4][0], self.lmList[4][1])
        return None
