# ch4 영역1 교안 3page
import cv2
import sys

# img=cv2.imread('apples.jpg')
img=cv2.imread('coins.png')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.blur(gray,(3,3))

circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,200,param1=150,param2=20,minRadius=50,maxRadius=120)    # 허프 원 검출 ---②
# circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,80,param1=150,param2=20,minRadius=20,maxRadius=40)
# 범위가 줄어들면 원이 작아짐

# method : 검출방법, HOUGH_GRADIENT
# dp : 이미지해상도 : accumulator해상도, 1이면 두 해상도 같음
# dist : 검출된 원 중심 사이의 최소 거리
# param1 : canny의 높은 threshold
# param2 : 누적 threshold
# minRadius, maxRadius : 검출할 원 반지름 범위 (해당 범위 안에 있는 원 검출)

print(circles)

# 값이 없을 수도 있으니까 if문으로 먼저 확인
if circles is not None:     # for 문 사용 -> circle 그려
    for i in circles[0]:
        # 중심(a,b), 반지름
        cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)	# 검출된 원 그리기 ---③

cv2.imshow('Hough circles',img)

cv2.waitKey()
cv2.destroyAllWindows()