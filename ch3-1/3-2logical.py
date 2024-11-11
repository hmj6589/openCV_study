import cv2
import sys
import numpy as np

src1=cv2.imread('lenna512.png') # 레나 이미지
src2=cv2.imread('opencv_logo256.png')  # 로고 이미지
# 현재 레나 / 로고 이미지가 크기가 다른 상태

if src1 is None or src2 is None:
    sys.exit('파일을 찾을 수 없습니다.')

mask = cv2.imread('opencv_logo256_mask.png', cv2.IMREAD_GRAYSCALE)   # 배경 하얀색+검정 로고
mask_inv = cv2.imread('opencv_logo256_mask_inv.png', cv2.IMREAD_GRAYSCALE)   # 배경 검정색+하얀색 로고

sy, sx = 100,100    # roi가 시작하는 위치 (0,0)은 좌측 상단
rows,cols,channels = src2.shape # 로고 이미지의 세로, 가로 크기 가져오기
roi = src1[sy:sy+rows, sx:sx+cols]  # 레나 이미지 잘라내기 (로고 크기와 동일하도록)
# cv2.imshow('roi', roi)

src1_bg = cv2.bitwise_and(roi, roi, mask=mask) # mask의 흰색(1)에 해당하는 roi는 그대로, 검정색(0)은 검정색으로
# cv2.imshow('src1_bg', src1_bg)

src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv) # mask_inv의 흰색(1)에 해당하는 src2는 그대로, 검정색(0)은 검정색으로
# cv2.imshow('src2_fg', src2_fg)

dst = cv2.bitwise_or(src1_bg, src2_fg)
# cv2.imshow('dst', dst)

src1[sy:sy+rows, sx:sx+cols] = dst

pp=np.hstack((src1_bg,src2_fg, dst))
cv2.imshow('point processing - logical',pp)
cv2.imshow('combine', src1)

cv2.waitKey()
cv2.destroyAllWindows()