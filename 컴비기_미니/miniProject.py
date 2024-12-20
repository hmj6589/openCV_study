import cv2
import numpy as np
import pytesseract
from PIL import Image
import sys
import re

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 전처리 함수
def preprocess_image(image):
    # BGR 컬러 영상을 명암 영상으로 변환하여 저장
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러 적용
    blur = cv2.blur(gray, (5,5))
    # Sobel 연산 : 수평 방향의 에지 검출
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)

    # 이진화
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7,17), np.uint8)
    # 모폴로지 연산
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    return morph

def find_candidates(image):
    # 이진화된 이미지에서 외곽선 찾기
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.minAreaRect : 찾은 외곽선을 감싸는 최소 면적의 회전된 직사각형 구하기
    rects = [cv2.minAreaRect(c) for c in contours]

    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates

# 종횡비와 넓이 조건
def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0:
        return False
    area = w * h
    aspect = w / h if w > h else h / w
    return 3000 < area < 12000 and 2.5 < aspect < 6.5

# Warp 변환 함수 (기울기 보정 및 직사각형으로 변환)
def rotate_plate(image, rect):
    # rect는 중심점, 크기, 회전 각도으로 표시
    center, (w, h), angle = rect

    w = w + 10
    h = h + 10

    if w < h:   # 세로가 긴 영역이면
        w, h = h, w     # 가로와 세로 맞바꿈
        angle -= 90     # 회전 각도 조정

    size = image.shape[1::-1]   # 행태와 크기는 역순
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)     # 회전 행렬 계산
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)   # 회전 변환

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)   # 후보영역 가져오기
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    _, warped_bin = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # 이진화

    # 외곽 배경 줄이기
    contours, _ = cv2.findContours(warped_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 가장 큰 컨투어 찾기
        largest_contour = max(contours, key=cv2.contourArea)

        # 해당 컨투어의 외곽 사각형 추출
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 번호판 영역 자르기
        warped_bin = warped_bin[y:y + h, x:x + w]

    # 색상 반전
    warped_inverted = cv2.bitwise_not(warped_bin)

    return cv2.resize(warped_inverted, (288, 56))  # pytesseract로 인식하기 적합한 크기로 조절

def ocr_plate(warped_plate):
    img_pil = Image.fromarray(warped_plate)

    # 번호판 이미지를 텍스트로 변환
    text = pytesseract.image_to_string(img_pil, lang='kor', config='--psm 7')
    return text.strip()

car_no = str(input("자동차 영상 번호 (00~09): "))
img = cv2.imread('cars/' + car_no + '.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 전처리
preprocessed = preprocess_image(img)

# 번호판 후보 영역 검출
candidates = find_candidates(preprocessed)

if not candidates:
    print("번호판 후보를 찾을 수 없습니다.")
else:
    img2 = img.copy()
    for i, candidate in enumerate(candidates):
        # 후보 영역
        pts = np.int32(cv2.boxPoints(candidate))

        # 후보 영역을 직사각형으로 변환하기
        warped = rotate_plate(img, candidate)

        # 변환된 이미지에서 텍스트 인식
        recognized_text = ocr_plate(warped)
        recognized_text = re.sub(r'[^0-9가-힣]', '', recognized_text)  # 숫자와 한글만 남기기

        # 숫자와 글자 비율 비교 및 첫 번째 문자가 숫자인지 확인
        num_count = sum(char.isdigit() for char in recognized_text)
        letter_count = sum(char.isalpha() for char in recognized_text)

        if recognized_text:
            # 만약 첫 번째 문자가 숫자가 아니라면 출력 X
            if not recognized_text[0].isdigit():
                recognized_text = recognized_text[1:]

            # 만약 글자의 비율이 숫자의 비율보다 높다면 출력 X
            if recognized_text and num_count > letter_count:
                # 변환된 번호판 출력
                print(f'{recognized_text}')

                cv2.imshow(f'candidate_img {car_no}', warped)

cv2.waitKey(0)
cv2.destroyAllWindows()
