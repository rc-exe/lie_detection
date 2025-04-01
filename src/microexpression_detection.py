import cv2
import dlib
from fer import FER
import numpy as np
from collections import Counter

# Load models with error handling
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    emotion_detector = FER()
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Constants for better configuration
SAD_THRESHOLD = 0.3  # Minimum confidence score for sad detection
FRAME_SAMPLE_RATE = 5  # Process every 5th frame in video
MIN_FRAMES_FOR_DECISION = 10  # Minimum frames needed for video analysis

def get_landmarks(image, face):
    """Get facial landmarks with error handling"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        landmarks = predictor(gray, face)
        return landmarks
    except Exception as e:
        print(f"Error getting landmarks: {e}")
        return None

def analyze_sad_landmarks(landmarks):
    """Enhanced analysis of facial landmarks specific to sadness"""
    if not landmarks:
        return False
    
    # Get relevant points for sadness detection
    # Eyebrows (points 17-26)
    left_eyebrow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)]
    right_eyebrow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)]
    
    # Mouth corners (points 48 and 54)
    mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
    mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
    
    # Calculate eyebrow slope (sadness often raises inner eyebrows)
    left_eyebrow_slope = (left_eyebrow[-1][1] - left_eyebrow[0][1]) / (left_eyebrow[-1][0] - left_eyebrow[0][0])
    right_eyebrow_slope = (right_eyebrow[0][1] - right_eyebrow[-1][1]) / (right_eyebrow[-1][0] - right_eyebrow[0][0])
    
    # Calculate mouth features (sadness often turns down mouth corners)
    mouth_width = mouth_right[0] - mouth_left[0]
    mouth_height = (landmarks.part(57).y - landmarks.part(51).y)  # Vertical distance between lips
    
    # Sadness indicators
    eyebrow_raised = left_eyebrow_slope < -0.1 or right_eyebrow_slope < -0.1
    mouth_turned_down = (landmarks.part(48).y > landmarks.part(51).y + 5 and 
                         landmarks.part(54).y > landmarks.part(51).y + 5)
    
    return eyebrow_raised and mouth_turned_down

def detect_microexpression(image):
    """Enhanced emotion detection with multiple methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = detector(gray)
    
    emotion = "neutral"
    score = 0.0
    is_sad_landmarks = False
    
    for face in faces:
        # Get facial landmarks
        landmarks = get_landmarks(image, face)
        
        # Analyze with FER
        try:
            emotion_data = emotion_detector.detect_emotions(image)
            if emotion_data:
                emotion, score = max(emotion_data[0]['emotions'].items(), key=lambda x: x[1])
        except Exception as e:
            print(f"FER error: {e}")
        
        # Additional sadness check through landmarks
        if landmarks:
            is_sad_landmarks = analyze_sad_landmarks(landmarks)
            
            # Visual debugging
            for n in range(0, 68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # Combine results - prioritize landmark analysis when FER is uncertain
        if is_sad_landmarks and (emotion != 'sad' or score < SAD_THRESHOLD):
            emotion = 'sad'
            score = max(score, 0.7)  # Boost confidence when landmarks confirm
        
        # Display results
        cv2.putText(image, f"Emotion: {emotion} ({score:.2f})", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Landmarks Sad: {is_sad_landmarks}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return image, emotion, score

def detect_lie(filepath):
    """Enhanced lie detection with better video processing"""
    file_extension = filepath.rsplit('.', 1)[1].lower()
    
    if file_extension in {'jpg', 'jpeg', 'png'}:
        # Image processing
        image = cv2.imread(filepath)
        if image is None:
            return "Error: Could not read image file"
            
        result_image, emotion, score = detect_microexpression(image)
        cv2.imwrite('static/uploads/result.jpg', result_image)
        
        # Enhanced lie detection criteria
        if emotion == 'sad' and score > SAD_THRESHOLD:
            return "Potential lie detected (sadness)"
        elif emotion in ['angry', 'fear', 'disgust'] and score > 0.4:
            return "Potential lie detected (negative emotion)"
        else:
            return f"Dominant emotion: {emotion} (confidence: {score:.2f})"

    elif file_extension in {'mp4', 'avi', 'mov'}:
        # Video processing
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return "Error: Could not open video file"
            
        frame_count = 0
        emotions = []
        scores = []
        landmark_sad_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every FRAME_SAMPLE_RATE-th frame
            if frame_count % FRAME_SAMPLE_RATE == 0:
                result_frame, emotion, score = detect_microexpression(frame)
                emotions.append(emotion)
                scores.append(score)
                
                # Check for landmark-based sadness
                if emotion == 'sad' or 'sad' in str(result_frame):
                    landmark_sad_frames += 1
                
                # Optional: Save sample frames for debugging
                if frame_count == 0:
                    cv2.imwrite('static/uploads/first_frame.jpg', result_frame)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        # Analyze results
        if frame_count == 0:
            return "Error: No frames processed"
            
        # Get most common emotion
        emotion_counter = Counter(emotions)
        dominant_emotion = emotion_counter.most_common(1)[0][0]
        avg_score = np.mean(scores) if scores else 0
        
        # Enhanced video lie detection logic
        if (dominant_emotion == 'sad' and avg_score > SAD_THRESHOLD) or \
           (landmark_sad_frames > frame_count // 4):  # If >25% frames show landmark sadness
            return "Potential lie detected (consistent sadness)"
        elif any(e in ['angry', 'fear', 'disgust'] for e in emotions) and \
             sum(s > 0.4 for s in scores) > MIN_FRAMES_FOR_DECISION:
            return "Potential lie detected (consistent negative emotions)"
        else:
            return f"Dominant emotion: {dominant_emotion} (avg confidence: {avg_score:.2f})"

    else:
        return "Unsupported file type. Please upload an image (jpg, png) or video (mp4, avi, mov)."