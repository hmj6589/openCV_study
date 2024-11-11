import cv2
import numpy as np

# 1. 블랙 or 화이트 이진 영상으로 만들어주기
#gray=cv2.imread('morph.jpg',cv2.IMREAD_GRAYSCALE)
gray=cv2.imread('morph_j.png', cv2.IMREAD_GRAYSCALE)
# THRESH_BINARY + THRESH_OTSU -> b라고 하는 이진 영상
t,b=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 구조 요소 만드는 방법 1
se1=np.uint8([[0,0,1,0,0],			# 구조 요소 (5x5)
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]])
# 구조 요소 만드는 방법 2
se2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
k=1     # 반복 횟수

b_dilation=cv2.dilate(b,se2,iterations=k)	# 팽창
b_erosion=cv2.erode(b,se2,iterations=k)	# 침식

b_opening=cv2.dilate(cv2.erode(b,se2,iterations=k),se2,iterations=k)	# 열기(침식 -> 팽창)
# b_opening=cv2.morphologyEx(b,cv2.MORPH_OPEN, se2)     # 한번에 적용 가능

b_closing=cv2.erode(cv2.dilate(b,se2,iterations=k),se2,iterations=k)	# 닫기(팽창 -> 침식)
# b_closing=cv2.morphologyEx(b,cv2.MORPH_CLOSE, se2)

morphology=np.vstack((b,b_dilation,b_erosion,b_opening,b_closing))
cv2.imshow('Morphology',morphology)

cv2.waitKey()
cv2.destroyAllWindows()