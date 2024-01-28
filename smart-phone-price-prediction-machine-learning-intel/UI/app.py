import streamlit as st
import pickle
import numpy as np

#import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Smart Phone Price Predictor")

#brand
company = st.selectbox('Brand', df['name'].unique())

#ram
ram = st.selectbox('RAM (in GB)', [1, 2, 4, 6, 8, 12, 16])

#storage
rom = st.selectbox('storage', [1, 8, 16, 32, 64, 128, 256, 512])

#camera
cam = st.number_input('camera in pixels')

#cpu
processor = st.selectbox('processor', df['processor'].unique())

#os
os = st.selectbox('OS', df['os'].unique())

#screen_size
screen_size = st.number_input('screensize in inches')

#resolution
resolution = st.selectbox('Display', ['1080x2160', '720x1600', '1080x2412', '1440x3088', '1440x3216', '1440x3200', '1080x2400', '1080x2376', '1080x2448', '750x1334', '1170x2532', '1179x2556', '1290x2796', '1080x2340', '1080x2408', '720x1520', '720x1648', '1080x2460', '720x1612', '720x1280', '1260x2800', '1080x2376', '720x1640', '480x854', '480x960', '876x2142', '1080x2280', '1080x2300', '1080x2448', '480x800', '480x960', '720x1560', '828x1792', '1125x2436', '1170x2532', '1284x2778', '1125x2436', '1440x3088', '1080x1920', '720x1480', '720x1280', '540x960', '1080x2220', '1440x2960', '720x1480', '1200x1920', '1200x2000', '720x1284', '1440x2960', '480x800', '1440x2560', '720x1650', '1440x2200', '1080x2430', '1080x2376', '720x1544'])






if st.button('Predict Price'):
    ppi = None
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, ram, rom, cam, processor, os, screen_size, ppi])

    query = query.reshape(1, 8)

    st.title(np.exp(pipe.predict(query)))

