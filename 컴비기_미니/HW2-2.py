# HW#2-2 : 자동차 번호판 후보 검출하기
# • 아래와 같은 방법으로 자동차 번호판의 후보를 검출한다.
# ① 전처리(잡음 제거) : 스무딩 또는 블러링
# ② 숫자 부분의 주요 특징인 세로 에지 검출
# ③ 임계값을 이용한 검은 배경과 흰 에지 분리
# ④ 가로로 긴 구조요소를 이용한 여러 번의 닫힘(close)를 통해 흰 숫자 에지를 팽창한 후 원상태로 침식

# 하나의 이미지에서 자동차번호판 후보를 검출하는 알고리즘을 작성한다.
# 제공한 6개의 테스트 이미지에 대해 차례대로 실행한다.

import cv2
import os
import numpy as np

def img_detection(image_path):
    # 1. 전처리 (잡음 제거) : 스무딩 또는 블러링
    # 그레이 이미지, Gaussian 블러 적용
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 그레이 이미지
    blur = cv2.GaussianBlur(gray, (3,3), 1.0)

    # 2. 숫자 부분의 주요 특징인 세로 에지 검출
    # Prewitt
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 수직 필터
    prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x)  # 수직 에지 (수직 필터를 적용해서 나온 결과)
    prewitt_x = cv2.convertScaleAbs(prewitt_grad_x)  # 절대값을 취해 양수 영상으로 변환, 수직 에지 - Abs(절댓값 취하는거)

    # 3. 임계값을 이용한 검은 배경과 흰 에지 분리
    ret, thresh_img = cv2.threshold(prewitt_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 가로로 긴 구조요소를 이용한 여러 번의 닫힘(close)을 통해 흰 숫자 에지를 팽창한 후 원상태로 침식
    # 구조 요소 만들어주기
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (13,3))  # 가로로 긴 구조 요소 생성
    # 여러번 닫기 : 팽창 -> 침식
    bclosing = cv2.erode(cv2.dilate(thresh_img, se, iterations=3), se, iterations=3)


    # 결과 출력
    cv2.imshow('img_detection_gray', gray)
    cv2.imshow('img_detection_prewitt_x_color', prewitt_x)
    cv2.imshow('img_detection_thresh_img_color', thresh_img)
    cv2.imshow('img_detection_bclosing_img_color', bclosing)

    cv2.waitKey()
    cv2.destroyAllWindows()

image_paths = ['cars/00.jpg', 'cars/01.jpg', 'cars/02.jpg','cars/03.jpg', 'cars/04.jpg', 'cars/05.jpg']

for image_path in image_paths :
    img_detection(image_path)