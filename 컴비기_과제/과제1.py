import cv2
import sys
import numpy as np

img = np.ones((600,900,3), np.uint8) * 255     # 1.600*900 크기의 컬러 이미지를 만들어 흰색으로 칠하시오.

BrushSize = 5
Blue = (255,0,0)
Red = (0,0,255)
Yellow = (0,255,255)
Mint = (255,255,0)
Green = (0,255,0)


def draw(event,x,y,flags,param):    # 마우스 콜백 함수
    global ix, iy

    if event==cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_ALTKEY:   # 2.Alt 키와 마우스 왼쪽 버튼의 다운/업을 이용하여 직사각형을 그리시오.
            ix,iy=x,y

        elif flags & cv2.EVENT_FLAG_CTRLKEY:    # 4. Ctrl 키와 마우스 왼쪽 버튼의 다운/업을 이용하여 원을 그리시오.
            ix, iy=x,y


    elif event==cv2.EVENT_LBUTTONUP:
        if flags & cv2.EVENT_FLAG_ALTKEY:   # 2.Alt 키와 마우스 왼쪽 버튼의 다운/업을 이용하여 직사각형을 그리시오.
            cv2.rectangle(img,(ix,iy),(x,y),Mint,2)

        elif flags & cv2.EVENT_FLAG_CTRLKEY:    # 4. Ctrl 키와 마우스 왼쪽 버튼의 다운/업을 이용하여 원을 그리시오.
            radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
            cv2.circle(img, (ix, iy), radius, Mint, 2)


    elif event==cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_ALTKEY:   # 3.Alt 키와 마우스 오른쪽 버튼의 다운/업을 이용하여 내부가 칠해진 직사각형을 그리시오.
            ix,iy=x,y

        elif flags & cv2.EVENT_FLAG_CTRLKEY:    # 5. Ctrl 키와 마우스 오른쪽 버튼의 다운/업을 이용하여 내부가 칠해진 원을 그리시오.
            ix, iy=x,y

    elif event==cv2.EVENT_RBUTTONUP:
        if flags & cv2.EVENT_FLAG_ALTKEY:   # 3.Alt 키와 마우스 오른쪽 버튼의 다운/업을 이용하여 내부가 칠해진 직사각형을 그리시오.
            cv2.rectangle(img,(ix,iy),(x,y),Mint,cv2.FILLED)

        elif flags & cv2.EVENT_FLAG_CTRLKEY:  # 5. Ctrl 키와 마우스 오른쪽 버튼의 다운/업을 이용하여 내부가 칠해진 원을 그리시오.
            radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
            cv2.circle(img, (ix, iy), radius, Mint, cv2.FILLED)

    elif event==cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_LBUTTON:
        if flags & cv2.EVENT_FLAG_SHIFTKEY: # 8.마우스 왼쪽 버튼과 Shift 키를 누르면서 움직이면 초록색 원(반지름 5)이 따라 그려진다.
            cv2.circle(img, (x, y), BrushSize, Green, -1)

        if not (flags & cv2.EVENT_FLAG_ALTKEY) and not (flags & cv2.EVENT_FLAG_CTRLKEY) and not (flags & cv2.EVENT_FLAG_SHIFTKEY):
            cv2.circle(img, (x, y), BrushSize, Blue, -1)  # 6.마우스 왼쪽 버튼을 누르면서 움직이면 파란색 원(반지름 5)이 따라 그려진다.

    elif event==cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_RBUTTON:
        if flags & cv2.EVENT_FLAG_SHIFTKEY: # 9.마우스 오른쪽 버튼과 Shift 키를 누르면서 움직이면 노란색 원(반지름 5)이 따라 그려진다.
            cv2.circle(img, (x, y), BrushSize, Yellow, -1)

        if not (flags & cv2.EVENT_FLAG_ALTKEY) and not (flags & cv2.EVENT_FLAG_CTRLKEY) and not (flags & cv2.EVENT_FLAG_SHIFTKEY):
            cv2.circle(img, (x, y), BrushSize, Red, -1)  # 7.마우스 오른쪽 버튼을 누르면서 움직이면 빨간색 원(반지름 5)이 따라 그려진다.



    cv2.imshow('Drawing',img)   # 수정된 이미지를 다시 그림

cv2.namedWindow('Drawing')  #창 이름 지정
cv2.imshow('Drawing',img)

cv2.setMouseCallback('Drawing',draw)   # 마우스콜백(창 이름, 콜백 함수)

while(True):
    if cv2.waitKey(1)==ord('s'):
        cv2.imwrite('painting.png', img)    # 이미지 저장

    elif cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()     # 모든 창 닫기
        break