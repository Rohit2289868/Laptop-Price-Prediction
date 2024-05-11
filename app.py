import streamlit as st
import pickle
import numpy as np

# Load the model and DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight validation
weight = st.number_input('Weight of the Laptop(kg)')
if not (1 <= weight <= 4.5):
    st.warning("Please enter a laptop weight between 1 and 4.5 kg.")

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size validation
screen_size = st.number_input('Screen Size (Enter in the range of 11-17 Inches)')
if not (11 <= screen_size <= 17):
    st.warning("Please enter a screen size between 11 and 17 inches.")

# Resolution
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440'])

# CPU Brand
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU Brand
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if (hdd == 0 and ssd == 0):
    st.warning("Please select at least one option for HDD or SSD.")

if st.button('Predict Price'):
    # Check if weight, screen size, and storage option are selected correctly
    if not (1 <= weight <= 4.5):
        st.warning("Please enter a laptop weight between 1 and 4.5 kg.")
    elif not (11 <= screen_size <= 17):
        st.warning("Please enter a screen size between 11 and 17 inches.")
    elif hdd == 0 and ssd == 0:
        st.warning("Please select at least one option for HDD or SSD.")
    else:
        # Prepare query and perform prediction
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        # Perform prediction
        predicted_price = pipe.predict(query)
        st.title("The predicted price of this configuration is " + str(int(np.exp(predicted_price[0]))))
