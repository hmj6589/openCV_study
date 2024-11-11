import cv2
import numpy as np

img=cv2.imread('fish1.jpg')	# 영상 읽기
#img=cv2.imread('fish2.jpeg')	# 영상 읽기

# 각각의 픽셀이 배경인지 물체인지 저정하는 배열
# mask 배열의 크기 == 이미지 크기 -> 해당 이미지 위치에 해당되는 각각의 픽셀에 대해 배경/물체인지 정할 것임
# 일단 초기값은 zeros -> 배경
mask=np.zeros(img.shape[:2], np.uint8)    # 모든 화소를 0(cv.2GC_BGD) 배경으로 초기화
#mask[:,:]=cv2.GC_PR_BGD

# roi -> 일정 영역 지정
rect = cv2.selectROI(img)
# 초기화가 사각형으로 되어 GC_INIT_WITH_REC -> 이 사각형의 정보가 3번째 인자 rect로 들어감
# 인자가 GC_INIT_WITH_REC인 경우에는 4,5번째 인자 사용 X
cv2.grabCut(img,mask,rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
#3 rect : 범위 지정. cv2.GC_INIT_WITH_RECT 모드에서만 사용됨.
#6 : 반복 횟수
#7 mode: GrabCut 적용 방법

# 결과로 나온 마스크에서 배경, 아마도 배경에 해당되는 부분을 0으로 두고 나머지는 1로 두겠다
# if (mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD)이면  mask2=0, 아니면 1
mask2=np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')

grab=img*mask2[:,:,np.newaxis]  # if mask2가 0이면 0(블랙), 아니면(1이면) 자기 자신의 색상 그대로
# np.newaxis : 차원을 높여줌 2차원(mask는 2차원) -> 3차원(우리가 볼려는 것은 컬러 이미지)

# mask 만들기 (내가 찾고자 하는 객체를 찾아서 그 객체를 흰색으로 바꿔줌)
# grab=255*mask2[:,:,np.newaxis]

cv2.imshow('Grab cut image',grab)

cv2.waitKey()
cv2.destroyAllWindows()