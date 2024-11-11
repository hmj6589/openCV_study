import cv2
import numpy as np

img=cv2.imread('rose.png')
cv2.imshow('Original',img)

img=cv2.rectangle(img,(180,290),(220,330),(255,0,0),2)
patch=img[290:330,180:220,:]        # 일부 잘랐음
# cv2.imshow('patch', patch)


# patch0 = cv2.resize(patch, (480, 480))
# fx로 12만큼 확대 / fy로 12만큼 확대
patch1=cv2.resize(patch,dsize=(0,0),fx=12,fy=12,interpolation=cv2.INTER_NEAREST)    # 화질 떨어짐
patch2=cv2.resize(patch,dsize=(0,0),fx=12,fy=12,interpolation=cv2.INTER_LINEAR)     # 디폴트, 퀄리티 좋아
patch3=cv2.resize(patch,dsize=(0,0),fx=12,fy=12,interpolation=cv2.INTER_CUBIC)      # 리니어보다 느리지만 퀄리티 더 좋아

dst1=np.hstack((patch1,patch2,patch3))      # 수평으로 연결해서 보여줘
# cv2.imshow('Resize - zoomin',dst1)


# img_small0 = cv2.resize(img, dsize=(297, 198))
# fx로 0.25만큼 축소 / fy로 0.25만큼 축소 == 1/4로 축소
img_small1 = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
img_small2 = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)  # 디폴트 (리니어)
img_small3 = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

dst2=np.hstack((img_small1,img_small2,img_small3))
cv2.imshow('Resize - zoomout',dst2)

cv2.waitKey()
cv2.destroyAllWindows()