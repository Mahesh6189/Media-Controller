import cv2 
import mediapipe as mp
import pyautogui
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize global variables
cap = None
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)
prev = -1
start_time = time.time()

# Function to count fingers
def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 1

    return cnt 

# Function to process video frames
def process_frames():
    global cap, drawing, hand_obj, prev, start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hand_obj.process(rgb_frame)

        if res.multi_hand_landmarks:
            hand_keypoints = res.multi_hand_landmarks[0]
            cnt = count_fingers(hand_keypoints)

            if prev != cnt:
                if time.time() - start_time > 0.2:
                    if cnt == 1:
                        pyautogui.press("right")
                    elif cnt == 2:
                        pyautogui.press("left")
                    elif cnt == 3:
                        pyautogui.press("up")
                    elif cnt == 4:
                        pyautogui.press("down")
                    elif cnt == 5:
                        pyautogui.press("space")
                    prev = cnt
                    start_time = time.time()
            
            drawing.draw_landmarks(frame, hand_keypoints, hands.HAND_CONNECTIONS)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Route to serve video feed
@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main route to render index.html
@app.route('/')
def index():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Unable to open camera."
    return render_template('index.html')

# Route to terminate capture
@app.route('/terminate_capture')
def terminate_capture():
    global cap
    try:
        if cap is not None:
            cap.release()
            cap = None
        return 'Capture terminated successfully!'
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
