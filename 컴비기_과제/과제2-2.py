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

    # 이미지 불러오기
    img = cv2.imread(image_path)

    if img is None:
        print("이미지 로드 실패")
        return

    # 1. 전처리 (잡음 제거) : 스무딩 또는 블러링
    # 그레일스케일 변환 및 Gaussian 블러 적용
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 전처리 과정 : 그레이스케일 변환
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 숫자 부분의 주요 특징인 세로 에지 검출
    # Sobel 필터 적용
    sobel_grad_x = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)  # 수직 에지 검출

    # 3. 임계값을 이용한 검은 배경과 흰 에지 분리
    # 검정과 흰 색을 잘 나눌 수 있는 최적의 임계값을 찾아주는 오츄 알고리즘 사용
    ret, thresh_img = cv2.threshold(sobel_grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 가로로 긴 구조요소를 이용한 여러 번의 닫힘(close)을 통해 흰 숫자 에지를 팽창한 후 원상태로 침식
    # 이진 영상으로 만들어주기
    t, b = cv2.threshold(sobel_grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 구조 요소 만들어주기
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))  # 가로로 긴 구조 요소 생성
    # 여러번 닫기 : 팽창 -> 침식
    bclosing = cv2.morphologyEx(b, cv2.MORPH_CLOSE, se, iterations=5)


    # hstack으로 이어주기 위해서는 gray -> bgr 과정 필요
    sobel_grad_x_color = cv2.cvtColor(sobel_grad_x, cv2.COLOR_GRAY2BGR)  # 에지 검출 이미지 컬러 변환
    thresh_img_color = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)  # 이진화 이미지 컬러 변환
    bclosing_img_color = cv2.cvtColor(bclosing, cv2.COLOR_GRAY2BGR)  # 이진화 이미지 컬러 변환

    # 결과 출력
    # grad = np.hstack((blur, sobel_grad_x_color, thresh_img_color, bclosing_img_color))
    cv2.imshow('img_detection_gray', gray)
    cv2.imshow('img_detection_sobel_grad_x_color', sobel_grad_x_color)
    cv2.imshow('img_detection_thresh_img_color', thresh_img_color)
    cv2.imshow('img_detection_bclosing_img_color', bclosing_img_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_paths = ['cars/00.jpg', 'cars/01.jpg', 'cars/02.jpg','cars/03.jpg', 'cars/04.jpg', 'cars/05.jpg']

for image_path in image_paths :
    img_detection(image_path)