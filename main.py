"""
Real-time Sign Language Detection System
Uses MediaPipe for hand tracking and custom ML model for gesture recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque

class SignLanguageDetector:
    def __init__(self, model_path='model/sign_language_model.pkl'):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load trained model if exists
        self.model = None
        self.label_encoder = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.label_encoder = data['label_encoder']
        
        # For smoothing predictions
        self.prediction_buffer = deque(maxlen=5)
        
        # ASL alphabet mapping (for display)
        self.asl_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        # Scale by hand size
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        if max_dist > 0:
            normalized = normalized / max_dist
            
        return normalized.flatten()
    
    def predict_sign(self, landmarks):
        """Predict sign language gesture"""
        if self.model is None:
            return "No Model", 0.0
        
        # Normalize landmarks
        normalized = self.normalize_landmarks(landmarks)
        
        # Predict
        prediction = self.model.predict([normalized])[0]
        probabilities = self.model.predict_proba([normalized])[0]
        confidence = np.max(probabilities)
        
        # Decode label
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Add to buffer for smoothing
        self.prediction_buffer.append((label, confidence))
        
        # Get most common prediction from buffer
        if len(self.prediction_buffer) >= 3:
            labels = [p[0] for p in self.prediction_buffer]
            most_common = max(set(labels), key=labels.count)
            avg_confidence = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
            return most_common, avg_confidence
        
        return label, confidence
    
    def draw_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks on image"""
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        prediction_text = ""
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.draw_landmarks(frame, hand_landmarks)
                
                # Extract and predict
                landmarks = self.extract_landmarks(hand_landmarks)
                label, conf = self.predict_sign(landmarks)
                prediction_text = label
                confidence = conf
        
        return frame, prediction_text, confidence
    
    def run(self):
        """Run real-time detection"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Sign Language Detection Started!")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, prediction, confidence = self.process_frame(frame)
            
            # Draw UI
            self.draw_ui(processed_frame, prediction, confidence)
            
            # Display
            cv2.imshow('Sign Language Detection', processed_frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_ui(self, frame, prediction, confidence):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw title
        cv2.putText(frame, "ASL Sign Language Detection", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw prediction
        if prediction and confidence > 0.5:
            # Large prediction text
            cv2.putText(frame, f"Sign: {prediction}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Confidence bar
            bar_width = int(300 * confidence)
            cv2.rectangle(frame, (w - 320, 20), (w - 20 + bar_width, 50), (0, 255, 0), -1)
            cv2.rectangle(frame, (w - 320, 20), (w - 20, 50), (255, 255, 255), 2)
            cv2.putText(frame, f"{confidence*100:.1f}%", (w - 310, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Show a sign...", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # Draw instructions at bottom
        cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run()
