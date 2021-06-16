import pandas as pd
import numpy as np
import cv2

# Get the boxes for the objects detected by YOLO by running the YOLO model.

def yolo_v3(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.

    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'motorbike',
        5: 'bus',
        7: 'truck',
        9: 'trafficLight',
        11:'stop sign',
        15:'cat',
        16:'dog'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
        "cat": [255, 255, 166],
        "dog": [0, 255, 255],
        "stop sign": [100,100,100],
        "bus": [150,180,180]
    }

videoCapture = cv2.VideoCapture("F:\Applied Computer Version\Final\SK.mp4")

#获取帧率和大小
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#设置输出的视频信息（视频文件名，编解码器，帧率，大小）
videoWriter = cv2.VideoWriter(
    "myTestVideo.avi",cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)

#读取视频文件，如果要读取的视频还没有结束，那么success接收到的就是True，每一帧的图片信息保存在frame中，通过write方法写到指定文件中
success,frame = videoCapture.read()
while success:
    yolo_boxes = yolo_v3(frame, 0.8, 0.5)
    for _, (x_min, y_min, x_max, y_max, label) in yolo_boxes.iterrows():
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), LABEL_COLORS[label], 2)  # 框的左上角，框的右下角
        frame = cv2.putText(frame, label, (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX, 0.5, LABEL_COLORS[label],1)  # 框的左上角
    videoWriter.write(frame)
    success, frame = videoCapture.read()