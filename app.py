import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model & dataframe
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# ---------------------------------------------------------
# Page Styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
)

# FIX WHITE LINE AT TOP + REMOVE HEADER + REMOVE PADDING
st.markdown("""
<style>

html, body {
    margin: 0 !important;
    padding: 0 !important;
}

header { visibility: hidden; }

.block-container {
    padding-top: 0rem !important;
}

main .block-container {
    padding-top: 5px !important;
    margin-top: -40px !important;
}

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white !important;
}

/* Title */
.big-title {
    font-size: 45px !important;
    font-weight: 900 !important;
    text-align: center;
    color: #00eaff;
    margin-bottom: 30px;
    text-shadow: 2px 2px 15px #000;
}

/* Column Cards */
.stCard {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 30px rgba(0,0,0,0.3);
    backdrop-filter: blur(6px);
}

/* FIX LABEL VISIBILITY */
.stSelectbox label,
.stNumberInput label,
.stSlider label,
.stTextInput label {
    color: #e0e0e0 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Remove unwanted label background */
.st-emotion-cache-ue6h4q {
    background: transparent !important;
}

/* Prediction Box */
.predicted {
    font-size: 35px;
    font-weight: 700;
    color: #00ffab;
    text-align: center;
    margin-top: 20px;
    padding: 15px;
    background: rgba(0,0,0,0.35);
    border-radius: 10px;
}

</style>

""", unsafe_allow_html=True)

# Title
st.markdown("<div class='big-title'>💻 Laptop Price Predictor</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Layout divided into columns
# ---------------------------------------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 📦 Laptop Specifications")

    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)

        company = st.selectbox('Brand', df['Company'].unique())
        types = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
        weight = st.number_input('Weight (kg)', step=0.1)
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
        ips = st.selectbox('IPS Display', ['No', 'Yes'])
        screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

        resolution = st.selectbox(
            'Screen Resolution',
            ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
        )

        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### ⚙️ Hardware Details")

    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)

        cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
        hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
        ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
        gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
        os = st.selectbox('Operating System', df['os'].unique())

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction Button
# ---------------------------------------------------------
st.markdown("### 🔮 Predict Laptop Price")

predict_btn = st.button("⚡ Predict Now", use_container_width=True)

if predict_btn:

    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # PPI calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': types,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'Ips': ips,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    pred = np.exp(pipe.predict(query)[0])

    st.markdown(
        f"<div class='predicted'>Estimated Price: ₹ {int(pred):,}</div>",
        unsafe_allow_html=True
    )
