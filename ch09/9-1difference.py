import cv2
import sys

# 동영상에서 frame 읽는 것을 별도의 함수로 했어욥
def get_frame(cap):
    ret, frame = cap.read()
    # 만약 ret이 있다면 (동영상이 끝나지 않았다면) -> gray 이미지로 변환
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:   # 비디오의 끝
        return []

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture('slow_traffic_small.mp4')    # 비디오 파일

if not cap.isOpened():
    sys.exit('동영상 연결 실패')

# while 문 들어가기 전에 frame 읽기
prev_frame = get_frame(cap)
cur_frame = get_frame(cap)
next_frame = get_frame(cap)

while True:
    # abs가 앞에 붙는 것은 절댓값을 구하는 것
    # next_frame, cur_frame (다음 프레임과 현재 프레임의 차이를 구하는 것)
    # 값이 작으면 작을 수록 변화가 없다 -> 검정색으로 표시
    # 값이 크면 클수록 변화가 크다 -> 하얀색으로 표시
    diff = cv2.absdiff(next_frame, cur_frame)   # 차이를 구하는 것은 절댓값으로
    cv2.imshow('Difference in Video', diff)

    # while 문 들어갈 때마다 다음 frame으로 넘어가는
    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = get_frame(cap)     # get_frame을 통해서 새로운 프레임 가져오기

    # if next_frame == [] : # next_frame이 없으면
    #     print('프레임 획득에 실패하여 루프를 나갑니다.')
    #     break

    key = cv2.waitKey(1)  # 1밀리초 동안 키보드 입력 기다림(정수)
    if key == ord('q'):  # 'q' 키가 들어오면 루프를 빠져나감, ord()는 문자를 아스키 값으로 변환하는 함수
        break

cap.release()  # 카메라와 연결을 끊음
cv2.destroyAllWindows()

# 검은색으로 표현되는 것 : 배경
# 하얀색 : 움직이는 것(차이가 있는 것)