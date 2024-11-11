import cv2
import sys
import numpy as np

def shape_detect(c): # 매개변수 : 각 도형의 컨투어
    shape = "undefined" # 리턴값은 shape이라는 변수인데 문자열임! (어떤 도형인지)

    peri = cv2.arcLength(c, True) # 도형의 둘레 구하기
    # approxPolyDP : 사각형에서 대략의 도형을 구하는데 이용하는 함수 (여기서는 주어진 컨투어의 거리를 말함)
    # 앱실론 : 0.03 * peri -> 값이 크면 클 수록 적은 포인트로 대략의 도형 구성
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    #print(len(approx))

    # approxPolyDP에 포함되어 있는 점의 개수가 n개다 -> 도형 검출
    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4: # 점이 4개일 때는 다각형 (정사각형인지 직사각형인지 구별하는 작업 필요)

        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h) # w과 h의 비율 구하기 why? 이거 정사각형인지 아닌지 구별할려고

        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle" # 그 비율이 1에 가까울 때 정사각형

    elif len(approx) == 5:
        shape = "pentagon"

    elif len(approx) == 10:
        shape = "star"

    else:
        # 분모 : 둘레 제곱 / 분자 : 4파이 * 면적
        roundness = (4.0 * np.pi * cv2.contourArea(c)) / (peri * peri)
        # 라운드니스가 1에 가까울수록 원 !
        if roundness >= 0.85 and roundness <=1.15 :
            shape = "circle"

    return shape

# 이미지 읽어오기
img = cv2.imread('shapes1.png')
#img = cv2.imread('shapes2.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 이 이미지에서 각 도형에 대한 컨투어 찾을려고 함
h,w = img.shape[:2]
# 1. gray로 변환
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 2. canny
canny=cv2.Canny(gray,100,200)
contours,hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# 각각의 컨투어(각각 어떤 도형인지 처리)
for contour in contours:
    shape = shape_detect(contour)

    # 도형의 중심을 구하는 방법 - moment 이용
    m = cv2.moments(contour)
    area = m['m00']  # Contour 면적, cv.contourArea(contour)
    cx, cy = int(m['m10'] / area), int(m['m01'] / area) # m10 : x좌표 합 / m01 : y좌표 합
    #print(cx,cy,area)

    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    # 도형의 중심에 텍스트 표시
    cv2.putText(img, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.imshow("Image", img)

    k=cv2.waitKey()
    if k==ord('q'):
        cv2.destroyAllWindows()
        break