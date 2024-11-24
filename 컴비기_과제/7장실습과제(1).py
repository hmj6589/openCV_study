import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Face 분류기 로드

# 이미지 불러오기
img = cv2.imread('face_images/face (6).jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 축소 비율 설정
scale_factor = 0.5  # 이미지 크기를 50%로 축소
img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출

face_mask = cv2.imread('sticker_images/sticker_img (1).png')  # 가면 이미지
h_mask, w_mask = face_mask.shape[:2]

for (x, y, w, h) in faces:
    if h > 0 and w > 0:
        roi = img[y:y+h, x:x+w]  # 검출된 얼굴에 대한 사각형 영역을 관심 영역(ROI)로 설정
        cv2.imshow('ROI', roi)  # 검출된 얼굴 표시
        cv2.moveWindow('ROI', 300, 50)

        # 가면 이미지의 크기를 검출된 얼굴의 크기와 같도록 resize
        face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)

        # ① 가면 마스크에서 검지 않은 부분만 통과(검은색 부분은 투명)
        gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 220, 255, cv2.THRESH_BINARY_INV)
        masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

        # ② 검출된 얼굴에서 가면 마스크의 검은 부분만 통과(검지 않은 부분은 투명)
        mask_inv = cv2.bitwise_not(mask)
        masked_img = cv2.bitwise_and(roi, roi, mask=mask_inv)

        cv2.imshow('First Image', img)

        # 흰색 마스크와 마스크된 얼굴 합성 (① + ②)
        img[y:y+h, x:x+w] = cv2.add(masked_face, masked_img)

        cv2.imshow('Masked Face', masked_face)  # 마스크된 얼굴 표시
        cv2.moveWindow('Masked Face', 600, 50)
        cv2.imshow('Final Image', img)  # 최종 결과 표시

cv2.waitKey()
cv2.destroyAllWindows()
