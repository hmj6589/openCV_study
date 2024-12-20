import numpy as np
import cv2
import sys

cap=cv2.VideoCapture('slow_traffic_small.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')
    
ret, prev_frame=cap.read()		# 첫 프레임
prev_gray=cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)   # gray로 변환

while(1):
    ret,frame=cap.read()	# 비디오를 구성하는 프레임 획득
    if not ret:
        sys('프레임 획득에 실패하여 루프를 나갑니다.')

    # 여기에 대해서도 gray로 바꿔주기
    curr_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 이전에 해당되는 프레임, 현재 프레임을 옵션으로 넣어주기
    # 모든 픽셀에서의 움직임 값을 다 계산
    flow=cv2.calcOpticalFlowFarneback(prev_gray,curr_gray,None,0.5,3,15,3,5,1.2,0)    # optical flow 계산

    # 16픽셀 단위로
    for y in range(16//2,frame.shape[0],16):
        for x in range(16//2,frame.shape[1],16):
            dx,dy = flow[y,x].astype(int)
            if (dx*dx+dy*dy) > 1 :      # 그 값이 1보다 크면 움직임이 큰 것
                cv2.line(frame,(x,y),(x+dx,y+dy),(0,0,255),2) # 큰 모션 있는 곳은 빨간색
            else :
                cv2.line(frame,(x,y),(x+dx,y+dy),(0,255,0),2)   # 길이가 짧은 것은 초록색

    cv2.imshow('Optical flow',frame)

    prev_gray=curr_gray   # 현재 프레임을 이전 프레임으로

    key=cv2.waitKey(30)	# 30밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv2.destroyAllWindows()