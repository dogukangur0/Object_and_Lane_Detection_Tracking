import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

deepSort = DeepSort()
model = YOLO('yolo11n.pt')

def roi_points(height, width):
    bl = (int(width*0.15), int(height*0.9))
    br = (int(width*0.9), int(height*0.9))
    tr = (int(width*0.68), int(height*0.7))
    tl = (int(width*0.33), int(height*0.7))

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,height], [width, 0], [width, height]])
    return pts1, pts2

def birds_eye_view(pts1, pts2, frame, height, width):
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height), flags = cv2.INTER_LINEAR)
    return transformed_frame

def mask_processing(transformed_frame):
    hsv_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 25, 255])
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    lower_yellow = np.array([10, 90, 150])
    upper_yellow = np.array([70, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    lower_amber = np.array([10, 100, 90])
    upper_amber = np.array([25, 255, 255])
    mask_amber = cv2.inRange(hsv_frame, lower_amber, upper_amber)
    
    mask_all_yellow = cv2.bitwise_or(mask_yellow, mask_amber)
    mask = cv2.bitwise_or(mask_white, mask_all_yellow)

    return mask

def line_preprocessing(mask, pts1, pts2):
    gauss_image = cv2.GaussianBlur(mask, (5,5), 0)
    lines = cv2.HoughLinesP(gauss_image, 1, np.pi/180, 50, minLineLength=50, maxLineGap=30)
    m = cv2.getPerspectiveTransform(pts2, pts1)
    return lines, m

def object_tracking(frame):
    detections = []
    labels = []
    results = model(frame)
    classes = results[0].names
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            label = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            labels.append(label)
            detections.append([[x1, y1, x2-x1, y2-y1], confidence, label])
    tracks = deepSort.update_tracks(detections, frame = frame)

    for label, track in zip(labels,tracks):
        x, y, w, h = track.to_tlwh()

        cv2.rectangle(frame, (int(x),int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        cv2.putText(frame, classes[label], (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

video_path = 'video/project_video.mp4'
capture = cv2.VideoCapture(video_path)
while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (1280, 720))
    height, width = frame.shape[:2]
    if not ret:
        break
    pts1, pts2 = roi_points(height, width)
    transformed_frame = birds_eye_view(pts1, pts2, frame, height, width)
    mask = mask_processing(transformed_frame)
    lines, m = line_preprocessing(mask, pts1, pts2)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pts = np.float32([[x1,y1],[x2,y2]]).reshape(-1,1,2)
        original_pts = cv2.perspectiveTransform(pts, m)
        x1o, y1o = original_pts[0][0]
        x2o, y2o = original_pts[1][0]
        cv2.line(frame, (int(x1o), int(y1o)), (int(x2o), int(y2o)), (0, 255, 0), 3)

    object_tracking(frame)

    cv2.imshow('result', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


