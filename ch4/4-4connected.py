import cv2
import sys
import numpy as np

img = cv2.imread('coins.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray',gray)

median = cv2.medianBlur(gray, 3)
_, gray_bin = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Binary', gray_bin)

# binary gray 이미지 이용
#cnt, labels = cv2.connectedComponents(gray_bin)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_bin)

cv2.imshow('labelling', (labels*50).astype(np.uint8))

# label 특정 값(색상) 부여 가능
img[labels == 0] = [127,127,127]    # 블루
img[labels == 1] = [127,0,0]    # 그린
img[labels == 2] = [0,127,0]    # 레드
img[labels == 3] = [0,0,127]    # 옐로우
img[labels == 4] = [0,127,127]

for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
    (x, y, w, h, area) = stats[i]
    # 최근접 사각형의 좌측 상단, w, h, 면적
    print(i, stats[i])

    # 노이즈 제거
    if area < 20:
        continue

    # 2개는 같은 표현
    cv2.rectangle(img, (x, y, w, h), (255, 0, 255), 2)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

cv2.imshow('Connected components', img)

cv2.waitKey()
cv2.destroyAllWindows()