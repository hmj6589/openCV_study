import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Face 분류기 로드

# 비디오 가져오기
#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap=cv2.VideoCapture('face2.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

face_mask = cv2.imread('mask_fire.png')
h_mask, w_mask = face_mask.shape[:2]

while True:
    ret,img=cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    # 불러온 frame 그대로 쓰지 X -> gray로 변환 후 사용
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # face_cascade.detectMultiScale(이미지 변수, 스케일 요소, 얼굴 신뢰도)
    # 1.3 : 보통 얼굴의 1.3배 큰 얼굴까지 찾겠다
    # 신뢰도 : 얼굴에 최소한 5개의 후보 경계 박스가 있어야 해당 얼굴을 검출 (신뢰도 값이 높을 수록

    ksize = 15  # 모자익 3-19 참조
    # ksize가 클 수록 블러링 더 많이 됨

    for (x, y, w, h) in faces:  # 검출된 모든 얼굴에 대해
        if h>0 and w>0:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # 사각형으로 표시

            roi = img[y:y + h, x:x + w]  # 검출된 얼굴에 대한 사각형 영역을 관심 영역(ROI)로 설정


            # 가면 이미지의 크기를 검출된 얼굴의 크기와 같도록 resize
            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)


            # ① 가면 마스크에서 검지 않은 부분만 통과(검은색 부분은 투명)
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 220, 255, cv2.THRESH_BINARY_INV)
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

            # ② 검출된 얼굴에서 가면 마스크의 검은 부분만 통과(검지 않은 부분은 투명)
            mask_inv = cv2.bitwise_not(mask)
            masked_img = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # 흰색 마스크와 마스크된 얼굴 합성 (① + ②)
            img[y:y + h, x:x + w] = cv2.add(masked_face, masked_img)

    cv2.imshow('Face detection with haarcascade',img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()		# 카메라와 연결을 끊음
cv2.destroyAllWindows()