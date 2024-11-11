import cv2
import numpy as np

# 이진영상
gray=cv2.imread('morph_j.png', cv2.IMREAD_GRAYSCALE)
t,b=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# opencv skeletonize
size = np.size(b)
skeleton = np.zeros(b.shape,np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  # 구조 요소
done = False
while(not done):
    eroded = cv2.erode(b,element)   # 침식
    temp = cv2.dilate(eroded,element)   # 팽창 (침식 -> 팽창 == 오픈)
    # 오픈을 하면 화이트에 대한 노이즈가 제거되는데
    temp = cv2.subtract(b,temp)     # 제거된 노이즈가 무엇인가 찾기
    skeleton = cv2.bitwise_or(skeleton,temp)
    cv2.imshow('skeletonize', skeleton)

    cv2.waitKey()
    b = eroded.copy()

    zeros = size - cv2.countNonZero(b)
    if zeros==size:
        done = True

cv2.imshow('skeletonize', skeleton)

cv2.waitKey()
cv2.destroyAllWindows()