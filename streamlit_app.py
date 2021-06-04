import streamlit as st
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from PIL import Image
import cv2
from data import *

# Set page config
apptitle = 'Object Detection'
st.set_page_config(page_title=apptitle, page_icon=":face_with_monocle:")


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
        "bus": [150,180,180],
        "motorbike": [0,180,180]
    }

my_sender='1005943382@qq.com'    # 发件人邮箱账号
my_pass = 'tpgzthcmojtfbcfa'              # 发件人邮箱密码(当时申请smtp给的口令)
my_user='doublefishmmm@gmail.com'      # 收件人邮箱账号，我这边发送给自己
def mail():
    ret=True
    try:
        msg=MIMEText('There are pedestrians running the red light!!!','plain','utf-8')
        msg['From']=formataddr(["doublefish",my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To']=formataddr(["user",my_user])              # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject']="warning!"                # 邮件的主题，也可以说是标题

        server=smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender,[my_user,],msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()# 关闭连接
    except Exception:# 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret=False
    return ret


# Title the app
st.title('Object Detection :face_with_monocle:')

# define selectbox in the sidebar 
option = st.sidebar.selectbox(
    'Services you are interested',
    ('Introduction about this app', 'Choose the image you want to detect'))

if option == 'Introduction about this app':
    st.subheader('Introduction')
    '''
    Our Object Detection application can identify the image you upload and display it.

    The objects it can identify include traffic light :traffic_light: , car :car: , pedestrian :walking: , biker :bicyclist: .

    :point_left:Please select **Choose the image you want to detect** in the sidebar to start.

    You can see more about this app in the [github](https://github.com/YUA1024/YUA1024).
    '''


else:
    # set subheader
    st.subheader('Application')
    '''
    You can display the image in full size by hovering it and clicking the double arrow.
    '''
    # sidebar
    if st.sidebar.checkbox('Upload'):
        uploaded_file = st.sidebar.file_uploader("Choose a image you want to detect")
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", content_images_name)
        uploaded_file = content_images_dict[content_name]
        if st.sidebar.checkbox('jaywalkers'):
            col1, col2, col3 = st.beta_columns(3)
            with col1:
              st.header("A cat")
              st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

            with col2:
              st.header("A dog")
              st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)

            with col3:
              st.header("An owl")
              st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)


    
    if uploaded_file is not None:
        im = Image.open(uploaded_file)
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()
    # show the image in sidebar
    st.sidebar.image(im, caption="Input Image", width=256)
    
    # identify the image    
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    yolo_boxes = yolo_v3(im, 0.8, 0.5)
    pedstrain_exist = 0
    for _, (x_min, y_min, x_max, y_max, label) in yolo_boxes.iterrows():
        if pedstrain_exist == 0 and label is "pedestrian":
            pedstrain_exist = 1
        im = cv2.rectangle(im, (x_min, y_min), (x_max, y_max), LABEL_COLORS[label], 2)  # 框的左上角，框的右下角
        im = cv2.putText(im, label, (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX, 0.5, LABEL_COLORS[label],1)  # 框的左上角
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    Image.fromarray(np.uint8(im))
    
    # show the image in main page
    # if there are pedestrains on the road, warning and send email
    if pedstrain_exist:
        ret=mail()
        if ret:
            print("邮件发送成功")
        else:
            print("邮件发送失败")
        st.error('Exists Pedestrains!!!')
        
    st.image(im, caption='the image you choose', width=512)
    
    
