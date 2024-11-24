import cv2
import sys
import numpy as np

img=cv2.imread('london_street.png')
if img is None:
    sys.exit('파일이 없습니다.')
height,width=img.shape[0],img.shape[1]

# 20개의 클래스
class_names = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
          5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
          10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
          14: 'motorbike', 15: 'person', 16: 'pottedplant',
          17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔 다르게

# Loading ssd   (사전 학습된 모델)
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt','MobileNetSSD_deploy.caffemodel')

# Detect objects
# test할 이미지를 blob으로 바꾸기
# yolo인 경우에는 256으로 나눴는데 이거는 127.5로 나눔 -> 0에서부터 2사이의 숫자로 바꾸곘다
# size도 다름
blob = cv2.dnn.blobFromImage(img, 1/127.5, (300*width//height,300), (127.5,127.5,127.5), False)  #swapRB=True) # net에 입력되는 이미지를 blob 형식으로 변환
net.setInput(blob)      # blob 형태의 데이터를 net에 넣어줌
output = net.forward()  # net을 순방향으로 실행 -> forward : 검출된 결과 가지고 오겠다

for i in range(output.shape[2]):
    print(output[0, 0, i])  # _, class_id, confidence, x, y, x2, y2
    # [print 값의 의미]
    # 첫번째 값은 의미 X
    # 두번째 값 : 내가 인식한 객체 번호
    # 세번째 값 : 인식한 객체에 대한 인식률
    # 나머지 4개의 숫자 : (좌측)x,y,(우측)x,y
    # (이거 소수인데 why? 크기가 얼마든지 간에 주어진 크기만큼의 상대적인 위치를 0~1로 표현하겠다는 것)


    confidence = output[0, 0, i, 2]  # i번째 사물 정보의 결과값
    if confidence > 0.5 :  # 임계값(CONF_THR) 이상인 경우만
        class_id = int(output[0, 0, i, 1])
        # 3:7 : (좌측)x,y,(우측)x,y
        # 좌측 x랑 width 곱하고 좌측 y랑 height 곱하고... 이런 식
        x, y, x2, y2 = (output[0, 0, i, 3:7] * [width, height, width, height]).astype(int)

        text = str(class_names[class_id]) + '%.3f' % confidence
        cv2.rectangle(img, (x, y), (x2, y2), colors[class_id], 2)
        cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[class_id], 2)

cv2.imshow("Object detection by SSD",img)

cv2.waitKey()
cv2.destroyAllWindows()