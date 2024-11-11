import cv2
import numpy as np

color=cv2.imread('lenna256.png')

# Special effects
# 1. 양방향 필터(bilateralFilter) : 에지 성분 유지+가우시안 잡음 효과적으로 제거
bila = cv2.bilateralFilter(color, -1,10,5)

# 2. 에지 보존 필터
edgep=cv2.edgePreservingFilter(color, flags=1, sigma_s=60, sigma_r=0.4)

# 3. 카툰 효과 (수채화로 그린 효과)
sty=cv2.stylization(color,sigma_s=60, sigma_r=0.45)

# 4. 연필로 그린 효과 (리턴 값이 2개 - gray, color)
# 현재 gray는 1byte
graySketch, colorSketch = cv2.pencilSketch(color, sigma_s=60, sigma_r=0.7, shade_factor=0.02)

# 5. 유화로 그린 효과
oil=cv2.xphoto.oilPainting(color, 7, 1)
cv2.imshow('oilpainting', oi)

cgraySketch=cv2.cvtColor(graySketch,cv2.COLOR_GRAY2BGR)     # gray -> BGR 변환 (화면에 표현해주기 위해서 변환하는 것, 하나하나 보여줄 때는 변환할 필요 X)
special=np.hstack((color, bila, edgep, sty, cgraySketch, colorSketch, oil))
cv2.imshow('Special Effects',special)

cv2.waitKey()
cv2.destroyAllWindows()