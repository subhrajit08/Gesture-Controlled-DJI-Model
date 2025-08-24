import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from GestureModel import GestureModel
import time
from collections import deque

class LiveGestureDetector:
    def __init__(self, model_path='best_gesture_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = GestureModel().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load class names if available
            if 'classes' in checkpoint:
                self.classes = checkpoint['classes']
            else:
                self.classes = ["up", "down", "right", "left", "come", "stop", "turn", "blank"]
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Classes: {self.classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Image preprocessing
        self.transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),   # expand grayscale → 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # match 3 channels
])

        
        # For smoothing predictions
        self.prediction_buffer = deque(maxlen=7)
        self.confidence_threshold = 0.6
        
        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        
    def preprocess_roi(self, roi):
        """Apply the same preprocessing as in dataset collection"""
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        # Apply Gaussian blur
        blur = cv.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        
        return thresh
    
    def predict_gesture(self, roi):
        """Predict gesture from ROI"""
        try:
            # Preprocess ROI
            processed_roi = self.preprocess_roi(roi)
            
            # Convert to PIL Image and apply transforms
            roi_pil = Image.fromarray(processed_roi).convert('L')
            roi_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(roi_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.classes[predicted.item()]
                confidence_score = confidence.item()
                
                # Get all class probabilities for debugging
                all_probs = probabilities[0].cpu().numpy()
                
                return predicted_class, confidence_score, all_probs, processed_roi
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0, None, roi
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions using a buffer"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Get predictions with high confidence
        high_conf_predictions = [(pred, conf) for pred, conf in self.prediction_buffer 
                               if conf > self.confidence_threshold]
        
        if not high_conf_predictions:
            return prediction, confidence
        
        # Find most common prediction
        predictions_only = [pred for pred, _ in high_conf_predictions]
        most_common = max(set(predictions_only), key=predictions_only.count)
        
        # Average confidence for the most common prediction
        avg_confidence = np.mean([conf for pred, conf in high_conf_predictions 
                                if pred == most_common])
        
        return most_common, avg_confidence
    
    def draw_info_panel(self, frame, gesture, confidence, all_probs, fps):
        """Draw information panel on frame"""
        # Background for info panel
        overlay = frame.copy()
        cv.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Main prediction
        color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 255, 255)
        cv.putText(frame, f"Gesture: {gesture.upper()}", (20, 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv.putText(frame, f"Confidence: {confidence:.2f}", (20, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS
        cv.putText(frame, f"FPS: {fps:.1f}", (20, 100), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Top 3 predictions if available
        if all_probs is not None:
            top3_indices = np.argsort(all_probs)[-3:][::-1]
            y_offset = 130
            cv.putText(frame, "Top predictions:", (20, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, idx in enumerate(top3_indices):
                class_name = self.classes[idx]
                prob = all_probs[idx]
                cv.putText(frame, f"{i+1}. {class_name}: {prob:.3f}", 
                          (20, y_offset + 20 + i*15), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_confidence_bar(self, frame, confidence):
        """Draw confidence bar"""
        bar_x, bar_y = 450, 50
        bar_width, bar_height = 200, 20
        
        # Background bar
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                    (100, 100, 100), -1)
        
        # Confidence bar
        conf_width = int(confidence * bar_width)
        color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 255, 255)
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                    color, -1)
        
        # Border
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                    (255, 255, 255), 2)
        
        # Text
        cv.putText(frame, "Confidence", (bar_x, bar_y - 5), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self):
        """Main detection loop"""
        cap = cv.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*50)
        print("GESTURE RECOGNITION - LIVE DETECTION")
        print("="*50)
        print("Instructions:")
        print("- Place your hand in the green rectangle")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset prediction buffer")
        print("- Press 's' to save current frame")
        print("="*50 + "\n")
        
        frame_count = 0
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Define ROI (Region of Interest)
            roi_x, roi_y, roi_w, roi_h = 50, 50, 300, 300
            
            # Draw ROI rectangle
            cv.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 
                        (0, 255, 0), 2)
            
            # Extract ROI
            roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            # Predict gesture
            gesture, confidence, all_probs, processed_roi = self.predict_gesture(roi)
            
            # Apply smoothing
            smooth_gesture, smooth_confidence = self.smooth_predictions(gesture, confidence)
            
            # Calculate FPS
            end_time = time.time()
            frame_time = end_time - start_time
            if frame_time > 0:
                fps = 1.0 / frame_time
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)
            else:
                avg_fps = 0
            
            # Draw information
            self.draw_info_panel(frame, smooth_gesture, smooth_confidence, all_probs, avg_fps)
            self.draw_confidence_bar(frame, smooth_confidence)
            
            # Draw ROI label
            cv.putText(frame, "Place hand here", (roi_x, roi_y - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frames
            cv.imshow('Gesture Recognition', frame)
            
            # Show processed ROI in separate window
            if processed_roi is not None:
                processed_display = cv.resize(processed_roi, (200, 200))
                cv.imshow('Processed ROI', processed_display)
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.prediction_buffer.clear()
                print("Prediction buffer reset")
            elif key == ord('s'):
                filename = f"gesture_frame_{frame_count}.jpg"
                cv.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        cap.release()
        cv.destroyAllWindows()
        print("Detection stopped.")

if __name__ == "__main__":
    try:
        detector = LiveGestureDetector('best_gesture_model.pth')
        detector.run_detection()
    except FileNotFoundError:
        print("Error: Model file 'best_gesture_model.pth' not found!")
        print("Please train the model first by running 'python train.py'")
    except Exception as e:
        print(f"Error: {e}")




