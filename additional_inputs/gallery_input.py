import cv2
from deepface import DeepFace
import pyautogui
import time

def emotion_recognition_from_webcam():
    # Initialize webcam. Use 0 for the default webcam
    video = cv2.VideoCapture(0)

    last_typed_time = time.time()  # Track when we last typed a message

    while True:
        ret, frame = video.read()

        if not ret:
            break

        try:
            # Detect faces and analyze emotions
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            for face_info in result:
                x, y, w, h = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']
                emotions = face_info['emotion']

                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Annotate the dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                text = f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}%"

                # Check if 1 minute has passed since the last typing
                current_time = time.time()
                if current_time - last_typed_time >= 60:
                    pyautogui.typewrite(text)
                    pyautogui.press('enter')
                    last_typed_time = current_time  # Update the last typed time

                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error: {e}")

        # Display the processed frame
        cv2.imshow('Emotion Recognition', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the display window
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    time.sleep(10)
    emotion_recognition_from_webcam()
