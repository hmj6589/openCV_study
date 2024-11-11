# [HW2-1]
# 하나의 윈도우 창에 비디오를 보여주면서 키보드에 따라 다양한 특수효과 적용한다. 처음 시작은 기본값인 original로 한다.
# 매 프레임마다 현재의 특수효과를 적용한 후 키보드에 따라 다음에 적용할 특수효과를 결정한다.

# ① 비디오가 play되는 동안 키보드를 누르면 해당되는 특수효과를 적용한다.
# • ‘n’ : original, ‘b’ : bilateral, ‘s’ : stylization, ‘g’ : Gray pencilSketch, ‘c’ : Color pencilSketch, ‘o’ : oilPainting
# ② 해당되는 특수효과를 프레임에 글자로 나타낸다.

import cv2
import sys
import numpy as np

video = cv2.VideoCapture('rize_video.mp4')   # 비디오 파일 열기

if not video.isOpened():    # 비디오 파일이 열리지 않으면 오류 메시지 출력
    print('동영상 연결 실패: 경로 또는 파일 문제')
    sys.exit()

# 기본값은 original
current_effect = 'n'

while True:
    ret, frame = video.read()  # 비디오를 구성하는 프레임 획득

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    key = cv2.waitKey(1)

    # 키에 따라 현재 적용할 효과 결정
    if key == ord('n'):
        current_effect = 'n'
    elif key == ord('b'):
        current_effect = 'b'
    elif key == ord('s'):
        current_effect = 's'
    elif key == ord('g'):
        current_effect = 'g'
    elif key == ord('c'):
        current_effect = 'c'
    elif key == ord('o'):
        current_effect = 'o'
    elif key == ord('q'):
        break

    # 현재 선택된 효과 적용
    if current_effect == 'n':  # 원본
        effect_frame = frame
        cv2.putText(effect_frame, "Original", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    elif current_effect == 'b':  # 양방향 필터
        effect_frame = cv2.bilateralFilter(frame, 9, 75, 75)
        cv2.putText(effect_frame, "Bilateral", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    elif current_effect == 's':  # 스타일화
        effect_frame = cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
        cv2.putText(effect_frame, "Stylization", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    elif current_effect == 'g':  # 그레이 연필 스케치
        effect_frame, ColorPencilSketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
        cv2.putText(effect_frame, "Gray Pencil Sketch", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (128,128,128), 2)

    elif current_effect == 'c':  # 컬러 연필 스케치
        GrayPencilSketch, effect_frame = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
        cv2.putText(effect_frame, "Color Pencil Sketch", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    elif current_effect == 'o':  # 유화 효과
        effect_frame = cv2.xphoto.oilPainting(frame, 7, 1)
        cv2.putText(effect_frame, "Oil Painting", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    # 프레임 출력
    cv2.imshow('Video Effects', effect_frame)

# 비디오 캡처 해제 및 모든 창 닫기
video.release()
cv2.destroyAllWindows()
