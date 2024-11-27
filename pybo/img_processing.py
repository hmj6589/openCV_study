import cv2
import numpy as np

def embossing(img):
    femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray16 = np.int16(gray)
    embossing = np.uint8(np.clip(cv2.filter2D(gray16, -1, femboss) + 128, 0, 255))

    return embossing


# haar 코드는 대부분 cv2로 이루어져있지만, haarcascade_frontalface_default.xml 파일 별도로 필요함
# 그래서 flask에서 접근 가능한 공간에 이 파일이 있어야 함
# img, haarcascade_frontalface_default.xml 주소 알려줌
def haar_face(img, dir): # 이미지에서 얼굴 검색할려고 함
    face_cascade = cv2.CascadeClassifier(dir+'haarcascade_frontalface_default.xml')  # Face 분류기 로드

    # 불러온 frame 그대로 쓰지 X -> gray로 변환 후 사용
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # 얼굴 검출

    for (x, y, w, h) in faces:  # 검출된 모든 얼굴에 대해
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형으로 표시

    # imshow 안해도 됨(서버에서 창을 띄울 필요는 없잖아)
    return img

# yolo_v3을 위한 함수 2개
def construct_yolo_v3(dir):  # 욜로 학습 모델을 가지고 와서 구성하는 함수
    f = open(dir+'coco_names.txt', 'r')

    # 한 줄 씩 읽어서 calss_names(배열)에 저장
    class_names = [line.strip() for line in f.readlines()]

    model = cv2.dnn.readNet(dir+'yolov3.weights', dir+'yolov3.cfg')

    # model의 층
    layer_names = model.getLayerNames()

    # 마지막에 연결이 안된 층(출력하고자 하는 층) -> 아웃풋 층에 넣기
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    # 신경망의 출력을 담당하는 3개의 층, yolo_82, yolo_94, yolo_106

    return model, out_layers, class_names


# 객체 인식을 하겠다는 함수
def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    # 이미지 바로 넣을 수 없고, blob 거쳐서 넣어야 함
    test_img = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)

    yolo_model.setInput(test_img)  # 테스트 이미지를 YOLO 학습모델의 새 입력 값으로 설정
    output3 = yolo_model.forward(out_layers)  # YOLO 학습모델의 out_layers로 출력을 계산하기 위해 정방향으로 전달
    # output3 객체는 14*14*3*85 텐서, 28*28*3*85 텐서, 56*56*3*85 텐서를 리스트

    box, conf, id = [], [], []  # 박스, 신뢰도, 부류 번호
    for output in output3:  # 서로 다른 격자에 대해서 아웃풋 출력(격자는 총 3개)
        print(len(output))
        for vec85 in output:
            scores = vec85[5:]  # p1,p2,⋯,p80, 80개 부류의 인식률

            class_id = np.argmax(scores)  # 가장 큰 인식률의 부류
            confidence = scores[class_id]  # 가장 큰 인식률

            if confidence > 0.5:  # 신뢰도가 50% 이상인 경우만 취함 (0.5 이상인 애들만 !)
                print(vec85)  # <2>
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)  # [0~1]표현을 이미지 내 실제 객체 중심위치로 변환
                w, h = int(vec85[2] * width), int(vec85[3] * height)  # [0~1]표현을 이미지 내 실제 객체 크기로 변환
                x, y = int(centerx - w / 2), int(centery - h / 2)  # 객체의 좌측 상단 위치
                box.append([x, y, x + w, y + h])  # 위치
                conf.append(float(confidence))  # 확률값
                id.append(class_id)  # 그래서 최대값을 가지고 있는 애는 누구인가

    ind = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)  # 비최대억제(NonMaximum Suppression) 알고리즘을 적용  # <3> 0.4 -> 1.0
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects

def yolo_v3(img, dir):
    model, out_layers, class_names = construct_yolo_v3(dir)  # YOLO 모델 생성
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔 다르게

    # res : 각 격자에 대해서 검출된 객체, 겹친 사각형을 모두 제거한 이미지
    res = yolo_detect(img, model, out_layers)  # YOLO 모델로 객체 검출
    print(len(res))  # 검색된 객체 개수는 얼마인지 출력

    for i in range(len(res)):  # 검출된 물체를 영상에 표시
        x1, y1, x2, y2, confidence, id = res[i]
        text = str(class_names[id]) + '%.3f' % confidence
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
        cv2.putText(img, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

    return img

def mask_rcnn(img, dir):
    height, width = img.shape[0], img.shape[1]

    f = open(dir+'object_detection_classes_coco.txt', 'r')  # 90개 객체 검출
    class_names = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔 다르게

    net = cv2.dnn.readNetFromTensorflow(dir+"frozen_inference_graph.pb", dir+"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

    # Detect objects
    # 이미지 그냥 넣으면 안됨 -- blob
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)

    # forward 결과 2개임
    # detection_out_final : 검출된 값(box) / detection_masks : 분할된 값(masks)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])

    for i in range(boxes.shape[2]):
        box = boxes[0, 0, i]
        print(box)
        class_id = int(box[1])
        confidence = box[2]
        if confidence < 0.5:
            continue
        # Get box Coordinates
        # 좌측 상단 점
        x = int(box[3] * width)
        y = int(box[4] * height)
        # 우측 하단 점
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        # 1 detection : boxes
        text = str(class_names[class_id]) + '%.3f' % confidence
        # cv2.rectangle(img, (x, y), (x2, y2), colors[class_id], 2)
        cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[class_id], 2)

        # 2 segmentation : masks
        # (mask를 이용해서 객체를 segmentation)
        roi = img[y: y2, x: x2]  # 객체 영역 roi
        roi_height, roi_width, _ = roi.shape

        # Get the mask
        mask = masks[i, class_id]   # mask 크기는 15*15이다
        mask = cv2.resize(mask, (roi_width, roi_height))

        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  # mask로 객체 영역 획득
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # 객체 영역의 윤곽선 획득
        # contours를 찾으면 좋은 점 : 그 안의 정보도 알 수 있음

        for cnt in contours:
            cv2.fillPoly(roi, [cnt], colors[class_id])  # 윤곽선 내부 확인 -> segmentation
            img[y: y2, x: x2] = roi

    return img