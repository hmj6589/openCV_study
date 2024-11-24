import cv2
import sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw     # 한글

def construct_yolo_v3():
    # f=open('coco_names.txt', 'r')
    f = open('coco_names_kor.txt', 'r', encoding='UTF-8')     # 이렇게 하면 잘 적용 안됨 why? 한글은 2~3바이트 필요 / 영어는 1바이트
    class_names=[line.strip() for line in f.readlines()]

    model=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv2.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()		    # YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔 다르게

#cap=cv2.VideoCapture(0,cv.CAP_DSHOW)
cap=cv2.VideoCapture('slow_traffic_small.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

font = ImageFont.truetype('fonts/gulim.ttc', 20)    # 한글 폰트에 대한 설정


while True:
    ret,frame=cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    # res : 각 격자에 대해서 검출된 객체, 겹친 사각형을 모두 제거한 이미지
    res=yolo_detect(frame,model,out_layers)

    # 프레임을 한글을 쓸 수 있는 형태로 변환
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    for i in range(len(res)):
        x1,y1,x2,y2,confidence,id=res[i]
        text=str(class_names[id])+'%.3f'%confidence
        # cv2.rectangle(frame,(x1,y1),(x2,y2),colors[id],2)
        draw.rectangle((x1,y1,x2,y2),outline=tuple(colors[id].astype(int)), width=2)    # 한글
        # cv2.putText(frame,text,(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,1.5,colors[id],2)
        draw.text((x1,y1+30), text, font=font, fill=tuple(colors[id].astype(int)), stroke_width=2)  # 한글

    frame = np.array(img_pil)

    cv2.imshow("Object detection from video by YOLO v.3",frame)
    
    key=cv2.waitKey(1)
    if key==ord('q'): break 
    
cap.release()		# 카메라와 연결을 끊음
cv2.destroyAllWindows()