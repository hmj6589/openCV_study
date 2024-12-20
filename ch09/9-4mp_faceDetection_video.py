import cv2
import mediapipe as mp


dice = cv2.imread('dice.png', cv2.IMREAD_UNCHANGED)
dice = cv2.resize(dice, dsize=(0,0), fx=0.1, fy=0.1)
d_w, d_h = dice.shape[1], dice.shape[0]

mp_face_detection=mp.solutions.face_detection
mp_drawing=mp.solutions.drawing_utils

face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)

#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('face2.mp4')

while True:
    ret,frame=cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    res=face_detection.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    if res.detections:
        for detection in res.detections:
            # mp_drawing.draw_detection(frame,detection)
            p = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

            x1 = int(p.x * frame.shape[1]) - d_w//2      # 눈 중앙에서 dice 이미지의 좌측 상단
            y1 = int(p.y * frame.shape[0]) - d_h//2

            x2 = x1 + d_w   # 우측 하단
            y2 = y1 + d_h

            if x1>0 and y1>0 and x2<frame.shape[1] and y2<frame.shape[0] :
                alpha = dice[:,:,3:0]/255

                # 원래 주어진 frame하고 dice 이미지하고 더하겠다
                frame[y1:y2, x1:x2] = (1-alpha)*frame[y1:y2, x1:x2] + alpha*dice[:,:,3]



    cv2.imshow('MediaPipe Face Detection from video',frame) # cv2.flip(frame,1))	# 좌우반전
    if cv2.waitKey(5)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()