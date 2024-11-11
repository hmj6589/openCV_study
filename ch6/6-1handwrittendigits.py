import numpy as np
import cv2
import matplotlib.pyplot as plt
import winsound
import tensorflow as tf # 설치

# 학습된 모델
model=tf.keras.models.load_model('lenet_trained.keras') # tf 버전 확인 : 2.17.0

def reset():
    global img

    # 흰색으로 만드는 배열
    img=np.ones((200,520,3),dtype=np.uint8)*255
    for i in range(5):  # 5개의 빨간색 선의 사각형
        cv2.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255))
    cv2.putText(img,'e:erase s:show r:recognition q:quit',(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)

def grab_numerals():
    numerals=[]
    for i in range(5):
        # 빨간색 사각형 안에 있는 걸 roi로 잡아
        roi=img[51:149,11+i*100:9+(i+1)*100,0]

        # 학습시킬 때의 이미지 크기와 동일하게 만들어줘야함(28*28)
        # 255 - : 255에서 빼는 것 -> 반전 이미지(화이트->검/검->화이트)
        roi=255-cv2.resize(roi,(28,28),interpolation=cv2.INTER_CUBIC)

        numerals.append(roi)  
    numerals=np.array(numerals)
    return numerals

def show():     # 내가 그린 글씨 모아 보기
    numerals=grab_numerals()
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i],cmap='gray')
        plt.xticks([]); plt.yticks([])
    plt.show()
    
def recognition():
    # 테스트 데이터
    numerals=grab_numerals()
    numerals=numerals.reshape(5,28,28,1)    # 2차원
    numerals=numerals.astype(np.float32)/255.0      # 0~1 사이의 수로 정규화시켜주는 작업 필요 ****

    # 예측
    # 이미 학습이 되어 있기 때문에 predict하는 과정만 있으면 됨
    res=model.predict(numerals)     # 학습 모델로 예측
    class_id=np.argmax(res,axis=1)  # 가장 큰 값을 같은 레이블

    for i in range(5):
        cv2.putText(img,str(class_id[i]),(50+i*100,180),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)

    winsound.Beep(1000,500)    
        
BrushSiz=4
LColor=(0,0,0)

def writing(event,x,y,flags,param):
    #  마우스 이벤트 콜백 함수
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),BrushSiz,LColor,-1)
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,(x,y),BrushSiz,LColor,-1)

reset()
cv2.namedWindow('Writing')
cv2.setMouseCallback('Writing',writing)

while(True):
    cv2.imshow('Writing',img)
    key=cv2.waitKey(1)
    if key==ord('e'):
        reset()
    elif key==ord('s'):
        show()        
    elif key==ord('r'):
        recognition()
    elif key==ord('q'):
        break
    
cv2.destroyAllWindows()