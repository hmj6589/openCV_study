# 가위바위보를 하는 손 동작을 포함하는 동영상 만들기
# 피부색을 이용하여 손 영역 검출하기
# ConvexHull, ConvexityDefect 이용하여 손 끝점 개수 찾기
# 손 끝점 개수 이용하여 가위,바위,보 정하기

# 결과 동영상(1분 이상)은 과제 등록에 제출
# 코드는 출력해서 제출

# 1분 입력 비디오를 처리하여 저장하면 결과 비디오가 1분이 안될 수도 있습니다.
# 입력 비디오의 길이를 늘리거나(2~3분), 처리시 while 문 내 cv2.waitKey의 매개값을 1 보다 큰값(예, 30)으로 수정할 수 있습니다.
# 결과 비디오에는 최종 결과(실제 비디오와 가위바위보 글자 표시)만 포함하면 됩니다.
# 손에 대한 이진 이미지는 inRange 함수에서 피부색 정보를 이용하여 구할 수 있습니다.
# 프레임에 여러 개의 피부색 영역(contour)이 존재한다면, 손의 영역만 찾을 수 있습니다.
# 예를 들어, 가장 둘레가 큰 영역(contour)만 처리할 수 있습니다.
# ConvexityDefect로 결함을 구했을 때 손가락이 펴져서 생긴 깊은 결함과 주먹쥔 손등에서의 얕은 결함이 포함될 수 있습니다.
# 각 결함의 distance가 일정 정도 이상되는 깊은 결함을 이용하여 손 끝점 개수를 올바르게 구할 수 있습니다.


import cv2
import numpy as np
import sys

# 피부색 HSV 범위
lower_skin = np.array([0, 30, 60], dtype=np.uint8)     # 피부색 HSV의 하한값
upper_skin = np.array([20, 150, 255], dtype=np.uint8)  # 피부색 HSV의 상한값

# 비디오 파일 열기
video = cv2.VideoCapture('RockScissorsPaper_Video.mp4')

# 비디오 파일이 열리지 않으면 오류 메시지 출력 후 종료하도록 함
if not video.isOpened():
    print('동영상 연결 실패')
    sys.exit()

# 비디오 출력 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('rock_scissors_paper_과제.avi', fourcc, 20.0, (640, 480))

# 프레임 반복 처리
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # 프레임 크기 조정
    frame = cv2.resize(frame, (640, 480))

    # 피부색 영역 검출
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # BRG 색상 -> HSV 색상 변환
    mask = cv2.inRange(hsv, lower_skin, upper_skin) # inRange 함수 -> 피부색 HSV 범위에 해당하는 영역을 이진화하여 mask 생성

    # 구조 요소 설정
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 잡음 제거 및 경계 찾기
    mask = cv2.dilate(mask, se, iterations=4) # 팽창 연산 적용
    mask = cv2.GaussianBlur(mask, (5, 5), 100) # 가우시안 블러 적용하여 경계 매끄럽게 만들기

    # 컨투어 검출
    # mask에서 윤곽(컨투어) 검출하기
    contours, _ = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        # 가장 큰 윤곽(컨투어) 찾기
        cnt = max(contours, key=cv2.contourArea)

        # 검출된 손 영역의 윤곽(컨투어)을 보라색 선으로 그리기
        cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)

        # 컨벡스 홀 및 결함 찾기
        hull_indices = cv2.convexHull(cnt, returnPoints=False)

        # 손가락 사이의 오목한 결함 검출
        defects = cv2.convexityDefects(cnt, hull_indices)

        # 손가락 끝점 개수 계산
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]): # 결함 반복 -> 손가락 끝 찾기
                s, e, f, d = defects[i, 0]
                # 리스트 형태로 저장
                start = cnt[s][0]
                end = cnt[e][0]
                far = cnt[f][0]

                # 깊은 결함 -> 손가락으로 간주
                if d > 10000:
                    finger_count += 1


        # finger_count -> 가위바위보 판별
        if finger_count == 0:
            result_text = "Rock"
        elif finger_count == 2: # 손가락 2개일 경우 -> Scissors로 판별
            result_text = "Scissors"
        elif finger_count >= 3: # 손가락 3개 이상일 경우 -> Paper로 판별
            result_text = "Paper"
        else: # Rock, Scissors, Paper가 아닐 때 -> Unknown
            result_text = "Unknown"

        # 결과 텍스트 표시
        cv2.putText(frame, result_text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2)

    # 결과 프레임 저장
    out.write(frame)
    cv2.imshow("Rock Scissors Paper", frame)

    # 키 입력 대기
    key = cv2.waitKey(30)  # 키보드 입력 기다림
    if key == ord('a'):  # 'a' 키가 들어오면 루프를 빠져나감
        break

# 자원 해제 및 모든 창 닫기
video.release()
out.release()
cv2.destroyAllWindows()


