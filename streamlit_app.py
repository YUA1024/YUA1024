import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# -- Set page config
apptitle = 'Cloud Classification'
st.set_page_config(page_title=apptitle, page_icon=":cloud:")

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return
    
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

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
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'
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
    }

# Title the app
st.title('Cloud Classification :cloud:')

option = st.sidebar.selectbox(
    'Services you are interested',
    ('Introduction about this app', 'Choose the image you want to detect'))

if option == 'Introduction about this app':
    st.subheader('Introduction')
    '''

    Our cloud detection application can identify the type of cloud you upload and display it.

    The clouds it can identify include cirrus, cumulus, cumulonimbus

    :point_left:Please select **Choose the image you want to detect** in the sidebar to start.

    You can see more about this app in the [github](https://github.com/YUA1024/YUA1024).
    '''


else:
    st.subheader('Application')
    '''
    You can display the image in full size by hovering it and clicking the double arrow.
    '''
    # sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a image you want to detect")
    if uploaded_file is not None:
        im = Image.open(uploaded_file)
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        yolo_boxes = yolo_v3(im, 0.8, 0.5)
        for _, (x_min, y_min, x_max, y_max, label) in yolo_boxes.iterrows():
            im = cv2.rectangle(im, (x_min, y_min), (x_max, y_max), LABEL_COLORS[label], 2)  # 框的左上角，框的右下角
            im = cv2.putText(im, label, (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX, 0.5, LABEL_COLORS[label],1)  # 框的左上角
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Image.fromarray(np.uint8(im))
    else:
        im = Image.open(r"./clouds/1111.jfif")
    st.sidebar.image(im, caption="Input Image", width=256)
    st.image(im, caption='the cloud you choose', width=800)

