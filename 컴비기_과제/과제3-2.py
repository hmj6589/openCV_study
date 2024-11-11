# ① 자동차 이미지 번호를 입력 받는다.
# ② HW 2-2를 이용하여 전처리를 한다.
# ③ 윤곽선(contours)을 찾아 최소면적 사각형을 찾는다.
# ④ 최소면적 사각형 중 가로 세로의 비율 등의 조건을 이용하여 자동차 번호판 후보를 찾는다.

import cv2
import numpy as np


# 주어진 사각형의 크기를 입력 받은 후
# 번호판 후보가 될 수 있을지 번호판 넓이와 종횡비 조건을 확인하는 함수
def verify_aspect_size(size):
    w, h = size # 사각형의 너비, 높이

    if h == 0 or w == 0:
        return False
    aspect = h / w if h > w else w / h  # 종횡비 계산(사각형의 가로와 세로 비율)

    # 번호판 넓이 조건과 종횡비 조건
    chk1 = 3000 < (h * w) < 12000       # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 8.0           # 종횡비 조건

    return chk1 and chk2


# 2. 전처리 함수
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Prewitt 필터를 이용한 세로 에지 검출
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x)
    prewitt_x = cv2.convertScaleAbs(prewitt_grad_x)

    # 이진화
    _, thresh_img = cv2.threshold(prewitt_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 가로로 긴 구조요소를 이용하여 닫힘 연산
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    bclosing = cv2.erode(cv2.dilate(thresh_img, se, iterations=3), se, iterations=4)

    return bclosing


# 3. 윤곽선 검출 및 최소 면적 사각형 표시
# 전처리된 이진화 이미지에서 윤곽선 검출
# -> 각 윤곽선에 최소 면적 사각형 그리기
def draw_min_area_rectangles(image, binary_image):
    # findContours -> 이진화 이미지에서 윤곽선 검출
    # RETR_EXTERNAL -> 외곽 윤곽선만 찾기
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_contours = image.copy()

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect) # 사각형의 네 꼭짓점 구하기
        box = np.int32(box)
        cv2.drawContours(img_contours, [box], 0, (0, 255, 255), 2)  # 모든 윤곽선에 대해 최소 면적 사각형 표시
    return img_contours, contours


# 4. 검출된 윤곽선 중 조건을 만족하는 번호판 후보(영역)만 표시
def draw_number_plate_candidates(image, contours):
    img_final = image.copy()
    found_candidate = False  # 번호판 후보가 발견되었는지 여부 추적

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        (x, y), (w, h), angle = rect

        # 번호판 후보 조건 확인
        # verify_aspect_size 함수 -> 사각형의 크기, 종횡비가 번호판 후보 조건에 만족하는지 확인
        if verify_aspect_size((w, h)):
            found_candidate = True
            cv2.drawContours(img_final, [box], 0, (0, 255, 0), 2)  # 조건을 만족하는 번호판 후보 표시

    if not found_candidate:
        print("번호판 후보 발견 X")
    return img_final


# 자동차 영상 번호 입력 -> 보여주기
car_no = str(input("자동차 영상 번호 (00~05): "))
img = cv2.imread('cars/' + car_no + '.jpg')

# 2. 전처리
binary_image = preprocess_image(img)
cv2.imshow("preprocess_image", binary_image)  # 2번째 사진

# 3. 윤곽선과 최소 면적 사각형 표시
img_contours, contours = draw_min_area_rectangles(img, binary_image)
cv2.imshow("draw_min_area_rectangles", img_contours)  # 3번째 사진

# 4. 조건을 만족하는 번호판 후보 표시
img_final = draw_number_plate_candidates(img, contours)
cv2.imshow("draw_number_plate_candidates", img_final)  # 4번째 사진

cv2.waitKey(0)
cv2.destroyAllWindows()
