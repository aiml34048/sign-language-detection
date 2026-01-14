"""
Train Sign Language Detection Model
Collects training data and trains a Random Forest classifier
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        self.data = []
        self.labels = []
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        if max_dist > 0:
            normalized = normalized / max_dist
            
        return normalized.flatten()
    
    def collect_samples(self, label, num_samples=100):
        """Collect training samples for a specific sign"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collected = 0
        print(f"\nCollecting {num_samples} samples for sign: {label}")
        print("Press SPACE to start collecting, 'q' to skip")
        
        collecting = False
        
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw UI
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {label} | Collected: {collected}/{num_samples}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if not collecting:
                cv2.putText(frame, "Press SPACE to start", (20, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Collecting... Show the sign!", (20, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if collecting:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        normalized = self.normalize_landmarks(landmarks)
                        self.data.append(normalized)
                        self.labels.append(label)
                        collected += 1
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                collecting = True
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {collected} samples for {label}")
    
    def collect_dataset(self, signs):
        """Collect dataset for multiple signs"""
        for sign in signs:
            self.collect_samples(sign, num_samples=100)
        
        print(f"\nTotal samples collected: {len(self.data)}")
        return np.array(self.data), np.array(self.labels)

def train_model(X, y):
    """Train Random Forest classifier"""
    print("\nTraining model...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_))
    
    return model, label_encoder

def save_model(model, label_encoder, path='model/sign_language_model.pkl'):
    """Save trained model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': label_encoder
        }, f)
    
    print(f"\nModel saved to {path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Sign Language Detection - Model Training")
    print("=" * 60)
    
    # Define signs to collect (ASL alphabet)
    signs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    # You can start with a subset for testing
    print("\nStarting with subset: A, B, C, D, E")
    signs = ['A', 'B', 'C', 'D', 'E']
    
    # Collect data
    collector = DataCollector()
    X, y = collector.collect_dataset(signs)
    
    if len(X) > 0:
        # Train model
        model, label_encoder = train_model(X, y)
        
        # Save model
        save_model(model, label_encoder)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("Run 'python main.py' to test the detector")
        print("=" * 60)
    else:
        print("No data collected. Exiting.")
