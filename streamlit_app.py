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
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    im = Image.open(uploaded_file)
else:
    im = Image.open(r"./clouds/1111.jfif")
scaler = int(im.height / 2)
st.sidebar.image(im, caption="Input Image", width=256)
#image = F.pil_to_tensor(im).float() / 255
