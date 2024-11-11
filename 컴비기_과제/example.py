import cv2
import numpy as np
import sys


cv2.imshow("example", 'RockScissorsPaper_Video.mp4')

# # 비디오 파일 열기
# video = cv2.VideoCapture('RockScissorsPaper_Video.mp4')
#
# # 비디오 출력 설정
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('rock_paper_scissors_output.avi', fourcc, 20.0, (640, 480))
#
# # 프레임 반복 처리
# while video.isOpened():
#     ret, frame = video.read()
#     if not ret:
#         break
#
#     # 프레임 크기 조정
#     frame = cv2.resize(frame, (640, 480))
#
#     cv2.imshow("Rock Scissors Paper", frame)