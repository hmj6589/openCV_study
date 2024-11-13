import cv2
import sys
import cvlib as cv
import numpy as np

img = cv2.imread('face2.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

faces, confidences = cv.detect_face(img)	 # 얼굴 검출

for (x, y, x2, y2) in faces:    # 검출된 모든 얼굴에 대해
    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2) # 사각형으로 표시

    # 얼굴 부분만 잘라내기
    face_img = img[y:y2, x:x2]                    # 얼굴 영역
    # detect_gender : 하나의 얼굴을 제공해 주면 그 얼굴로 인해서 성별 예측해 주는 함수
    label, g_confidence = cv.detect_gender(face_img)  # 성별 예측하기 : male, female
    print(g_confidence)
    print(label)
    gender = np.argmax(g_confidence)    # argmax : 제일 큰 값을 가지고 있는 애
    text = f'{label[gender]}:{g_confidence[gender]:.1%}'
    cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

cv2.imshow('CVLIB - gender detection',img)

cv2.waitKey()
cv2.destroyAllWindows()