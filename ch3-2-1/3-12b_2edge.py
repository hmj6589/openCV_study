import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(gray, -1)     # 라플라시안 필터 적용
cv2.imshow('Laplacian',laplacian)

# 샤프닝 필터와 비교
# 샤프닝 == 라플라시안
fsharpen=np.array([[0.0, 1.0, 0.0],
                  [1.0, -4.0, 1.0],
                  [ 0.0, 1.0, 0.0]])
# sharpen = 필터를 직접 적용하여 컨볼루션 한 것
sharpen = cv2.filter2D(gray, -1, fsharpen)

grad=np.hstack((gray, sharpen, laplacian))
cv2.imshow('Second derivatives',grad)

cv2.waitKey()
cv2.destroyAllWindows()