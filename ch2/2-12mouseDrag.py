import cv2
import sys

img=cv2.imread('girl_laughing.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

def draw(event,x,y,flags,param):        # 마우스 콜백 함수
    global ix,iy
    
    if event==cv2.EVENT_LBUTTONDOWN:	# 마우스 왼쪽 버튼 클릭했을 때 초기 위치 저장
        ix,iy=x,y
    elif event==cv2.EVENT_LBUTTONUP:	# 마우스 왼쪽 버튼 클릭했을 때 직사각형 그리기
        cv2.rectangle(img,(ix,iy),(x,y),(0,0,255),2)

    elif event==cv2.EVENT_RBUTTONDOWN:      # 마우스 오른쪽 버튼 클릭했을 때 초기 위치 저장
        ix, iy = x, y
    elif event==cv2.EVENT_RBUTTONUP:	# 마우스 오른쪽 버튼 클릭했을 때 직사각형 그리기
        cv2.rectangle(img, (ix, iy), (x, y), (255,0,0), 2)

    cv2.imshow('Drawing',img)

# 아래는 동일
cv2.namedWindow('Drawing') # 창 이름 지정
cv2.imshow('Drawing',img) # 창 띄우고 -> 보여주는

cv2.setMouseCallback('Drawing',draw) # 마우스콜백(창 이름, 콜백 함수)

while(True):
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        break