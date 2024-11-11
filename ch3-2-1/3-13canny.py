import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)

# Sobel 에지 : Canny 1~2 단계
# 소벨에 의한 에지 -> 에지가 너무 두꺼움
blur=cv2.GaussianBlur(gray,(3,3), 1.0)	# Canny 1단계 : (전처리과정) 노이즈 제거
sobel_grad_x = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)	# Canny 2단계 : 소벨 에지 검출
sobel_grad_y = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_grad_x)
sobel_y = cv2.convertScaleAbs(sobel_grad_y)
sobel_edge = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)


_, th = cv2.threshold(sobel_edge, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 오츄에 의해서 계산된 하나의 임계값 -> 이진화
# 하나의 임계 값을 이용하는 경우의 문제점
# 중간중간 약한 엣지들도 검출될 수도 있음
# 엣지가 두껍게 나올 수 있음

# Canny : 다양한 임계값
# Tlow보다 작은 에지가 아님 / Thigh보다 크면 에지가 무조건 맞음
canny1=cv2.Canny(gray,50,150)	# Tlow=50, Thigh=150으로 설정
canny2=cv2.Canny(gray,100,200)	# Tlow=100, Thigh=200으로 설정

canny=np.hstack((sobel_edge, th, canny1, canny2))
cv2.imshow('Canny',canny)

cv2.waitKey()
cv2.destroyAllWindows()