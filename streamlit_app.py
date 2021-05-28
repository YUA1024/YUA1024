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

st.title('Cloud detection')
st.write("Choose the image you want to detect!")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    im = Image.open(uploaded_file)
else:
    im = Image.open(r"./clouds/1111.jfif")
    
st.sidebar.image(im, caption="Input Image", width=256)
st.image(im, caption='the cloud you choose')

