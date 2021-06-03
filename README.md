Traffic Management System On Streamlit Apps
====

![maven](https://img.shields.io/badge/Python-3.6.5--3.9.0-green)
![maven](https://img.shields.io/badge/streamlit-0.82.0-yellow)
![maven](https://img.shields.io/badge/tensorflow-1.15.0-orange)
![maven](https://img.shields.io/badge/numpy-1.18.4-blue)
![maven](https://img.shields.io/badge/pandas-1.0.1-lightgrey)
![maven](https://img.shields.io/badge/opencv--python-4.5.2.52-yellowgreen)

This project is jointly completed by Zhao Zhiyuan, Li Yizhang and Wang Yifan

Background
-------

The project is a final work for Applied Computer Vision courses. We try to detect some objetcs and show them in some visible ways. We've already deployed this project in Streamlit Apps.   

Application
-------

1. Detect pedestrians on the highway and give warning. 
<div align=center><img width="600" height="450" src="https://github.com/YUA1024/YUA1024/blob/master/image/result1.png"/></div>
2. Identify vehicles and pedestrians which are running red lights and save their pictures.
<div align=center><img width="600" height="450" src="https://github.com/YUA1024/YUA1024/blob/master/image/result2.png"/></div>

Usage
-------

1. Go to the app website [https://share.streamlit.io/YUA1024/YUA1024/master/streamlit_app.py](https://share.streamlit.io/YUA1024/YUA1024/master/streamlit_app.py).
2. Choose the services you are interested in.
3. Choose the image you want to detect.
4. The page will display the recognition results and give feedback.

How to run this project
-------
```
pip install --upgrade streamlit opencv-python
streamlit run https://share.streamlit.io/YUA1024/YUA1024/master/streamlit_app.py  
```

Tool Landscape
-------

1. App Deployment
    Streamlit
2. Editor
    Pycharm
3. Model
    YOLOv3
4. CI/CD
    Github Actions
5. Language
    Python


Questions?Comments?
-------

Email us by doublefishmmm@gmail.com



