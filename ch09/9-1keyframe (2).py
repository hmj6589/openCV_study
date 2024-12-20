import cv2
import sys
import numpy as np

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        return None     


cap = cv2.VideoCapture('face2.mp4') #slow_traffic_small.mp4')    # 비디오 파일
if not cap.isOpened():
    sys.exit('동영상 연결 실패')

prev_frame = get_frame(cap)
cur_frame = get_frame(cap)
next_frame = get_frame(cap)

images=[]   # 여기에는 keyframe만 저장할려고 함

# next frame을 기준으로 !
images.append(cv2.resize(next_frame, dsize=(0, 0), fx=0.2, fy=0.2))

# 추가될 때마다 이미지 보여주기
cv2.imshow("keyframes", images[0])

while True:
    # absdiff로 차이 구하기
    diff = cv2.absdiff(next_frame, cur_frame)
    cv2.imshow('Difference in Video', diff)

    # 두 frame의 차이를 구해서 sum
    sum_diff = np.sum(diff)

    # 그 합이 20000000을 초과하면 (이 정도의 큰 차이를 발생하는 diff)에 대해서
    if sum_diff > 20000000:
        images.append(cv2.resize(next_frame, dsize=(0, 0), fx=0.2, fy=0.2))
        print(sum_diff)
        concat_image = np.hstack((images))      # 가로로 쭉 붙여서 보여주기
        cv2.imshow("keyframes", concat_image)

    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = get_frame(cap)

    if next_frame is None :    
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    key = cv2.waitKey(1)
    if key == ord('q'):  # 'q' 키가 들어오면 루프를 빠져나감, ord()는 문자를 아스키 값으로 변환하는 함수
        break

cv2.waitKey()
cap.release()  # 카메라와 연결을 끊음
cv2.destroyAllWindows()
