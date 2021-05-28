import streamlit as st
import numpy as np
import pandas as pd
#import kornia
#from torch import nn
#import torch
#from torchvision.transforms import functional as F
#from torchvision.utils import make_grid
#from streamlit_ace import st_ace
from PIL import Image

st.title('Cloud detection :cloud:')
'''
Our cloud detection application can identify the type of cloud you upload and display it

:point_left:Please select **Choose the image you want to detect** in the sidebar to start

You can display the image in full size by hovering it and clicking the double arrow
'''

# sidebar
uploaded_file = st.sidebar.file_uploader("Choose a image you want to detect")
if uploaded_file is not None:
    im = Image.open(uploaded_file)
else:
    im = Image.open(r"./clouds/1111.jfif")
st.sidebar.image(im, caption="Input Image", width=256)
st.sidebar.write("")

st.image(im, caption='the cloud you choose', width=1024)

