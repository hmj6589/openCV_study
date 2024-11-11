import cv2
import sys
import numpy as np

img=cv2.imread('star.png')
# img = cv2.imread('horse.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

h,w = img.shape[:2]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(gray,100,200)
cv2.imshow('contours-canny',canny)

contours,hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

contour=contours[0]
cv2.drawContours(img,contour,-1,(255,0,255),2)
cv2.imshow('contours',img)

perimeter = cv2.arcLength(contour, closed=True)     # 테두리(둘레 길이)
print('perimeter=', perimeter)

area = cv2.contourArea(contour)     # 면적 계산
print('area=', area)

x, y, width, height = cv2.boundingRect(contour)     # contour를 둘러싸고 있는 최근접
#cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)

rect = cv2.minAreaRect(contour) # (중심, (가로,세로), 기울기)
box = cv2.boxPoints(rect)   # 4개의 꼭지점 좌표
box = np.int32(box)     # 정수 좌표
print("box",box)
cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

(x, y), radius = cv2.minEnclosingCircle(contour)
cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 0), 2)     # 최근접 원이 그려짐

poly10 = cv2.approxPolyDP(contour, epsilon=10, closed=True)
poly20 = cv2.approxPolyDP(contour, epsilon=20, closed=True)
poly50 = cv2.approxPolyDP(contour, epsilon=50, closed=True)
# epsilon은 그 거리가 크면 클수록 더 대략적인 형태의 도형이 나옴
# 2 : epsilon. 다각형의 직선과의 허용 거리. 값이 크면 저장되는 좌표점의 개수가 작아짐
# 3 : closed?(닫힌 다각형?)
# cv2.drawContours(img, [poly10], 0, (255, 0, 0), 2)
print(poly10.shape)     # 28개의 점만 가지고 연결하겠다
# cv2.drawContours(img, [poly20], 0, (0, 255, 0), 2)
print(poly20.shape)     # 20개의 점만 가지고 연결하겠다
# cv2.drawContours(img, [poly50], 0, (0, 0, 255), 2)
print(poly50.shape)     # 8개의 점만 가지고 연결하겠다

# pointPolygonTest : 영상 안의 모든 픽셀마다 윤곽선 내부에 포함되는지 확인
# for y in range(h):
#    for x in range(w):
#        if cv2.pointPolygonTest(contour, (x, y), False) > 0:     # 0보다 크다? -> 아 내부에 위치하는구나~
#            img[y, x] = (255, 200, 255)

hull = cv2.convexHull(contour,returnPoints=True)    # 인접하고 있지만 볼록 다각형으로 만들 수 있어용
cv2.drawContours(img, [hull], 0, (100, 100, 100), 4)
print(hull.shape)

hull2 = cv2.convexHull(contour, returnPoints=False)
# 그리는 것뿐만 아니라 결함된 부분까지 찾아서 리턴할거야 한다면 -> False로 두기
defects = cv2.convexityDefects(contour, hull2)      # 볼록 다각형에서 결함 부분 정보 출력해줌
# contour를 구성하고 있는 에지
print(defects)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    print(i,' : ', s,e,f,d)
    cv2.circle(img, tuple(contour[s][0]), 5, [128, 0, 255], -1)
    cv2.circle(img, tuple(contour[e][0]), 5, [255, 0, 128], -1)
    cv2.circle(img, tuple(contour[f][0]), 5, [128, 255, 0], -1)

cv2.imshow('contours',img)

cv2.waitKey()
cv2.destroyAllWindows()