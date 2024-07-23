import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request
from PIL import Image
from streamlit_option_menu import option_menu
from pymongo import MongoClient
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Koneksi ke MongoDB
client = MongoClient("mongodb+srv://regaarzula:YlDDs2OYHYOuuLPc@cluster0.nslprzn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['hydroponic_system']
collection = db['temperature_settings']

# URL ESP32 CAM
url = 'http://192.168.1.13/cam-hi.jpg'
image_placeholder = st.empty()

# Kelas objek untuk model YOLO
classNames = ["Immature Sawi", "Mature Sawi", "Non-Sawi", "Partially Mature Sawi", "Rotten"]

def process_frame(frame, model, min_confidence):
    results = model(frame)
    for detection in results[0].boxes.data:
        x0, y0 = (int(detection[0]), int(detection[1]))
        x1, y1 = (int(detection[2]), int(detection[3]))
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = classNames[cls]
        label = f'{object_name} {score}'

        if score > min_confidence:
            # Gambar bounding box
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # Hitung ukuran teks
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_width, label_height = label_size
            baseline = max(baseline, 1)

            # Definisikan posisi latar belakang kotak label
            label_x0 = x0
            label_y0 = y1 + 10
            label_x1 = x0 + label_width + 10
            label_y1 = label_y0 + label_height + baseline

            # Gambar kotak terisi sebagai latar belakang label
            cv2.rectangle(frame, (label_x0, label_y0 - label_height - 10), (label_x1, label_y1), (0, 0, 255), -1)

            # Gambar teks label
            cv2.putText(frame, label, (x0 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def detect_objects_in_image(model, uploaded_file, min_confidence):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    result_frame = process_frame(image_bgr, model, min_confidence)
    result_image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption='Processed Image', use_column_width=True)

def detect_objects_in_video(model, uploaded_file, min_confidence):
    video_file = uploaded_file.read()
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file)
        
    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 0)
        frame = process_frame(frame, model, min_confidence)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    st.write("Video processing completed.")

def main():
    # Sidebar navigation menggunakan option_menu
    with st.sidebar:
        menu_selection = option_menu(
            menu_title="Navigation Menu",
            options=["Home", "Monitoring", "Controlling", "Object Detection"],
            icons=["house", "clipboard-data-fill", "search"],
            menu_icon='cast',
            default_index=1,
            orientation="vertical"
        )

    if menu_selection == "Home":
        st.markdown("<h1 style='text-align: center;'>Hydroponic Tech House - IoT Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
            """
            Welcome to Hydroponic Tech House, where you can manage and monitor your hydroponic system smartly!
            Use the sidebar navigation on the left to explore our features.
            """
        )

    elif menu_selection == "Monitoring":
        st.markdown("<h1 style='text-align: center;'>Monitoring</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>View real-time sensor data and environmental conditions of your hydroponic system.</p>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        # Create a form for Ubidots widgets
        with st.form("monitoring_form"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🌡 Temperature")
                widget_url1 = "https://stem.ubidots.com/app/dashboards/public/widget/n2DJ6zraCJkvxZYYAQ5egHCTgZLe6E3XBpVtLGnZsoQ"
                st.components.v1.iframe(widget_url1, width=330, height=400, scrolling=True)

                st.subheader("🌬 Air Quality")
                widget_url2 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
                st.components.v1.iframe(widget_url2, width=330, height=400, scrolling=True)

            with col2:
                st.subheader("💧 Humidity")
                widget_url3 = "https://stem.ubidots.com/app/dashboards/public/widget/XXSQaCPoG41tQ1W33PDj9xphZOO7DwF6tvflxiKnSkE"
                st.components.v1.iframe(widget_url3, width=330, height=400, scrolling=True)

                st.subheader("🌱 Soil Moisture")
                widget_url4 = "https://stem.ubidots.com/app/dashboards/public/widget/ST57XPDVjOhWeqD1GHC1ejT2zCuxr078rU-tQH6WNKo"
                st.components.v1.iframe(widget_url4, width=330, height=400, scrolling=True)

            # Add form submit button
            submit_button = st.form_submit_button('')

    elif menu_selection == "Controlling":
        st.title('Pengaturan Suhu Minimum dan Maksimum')

        min_temp = st.number_input('Suhu Minimum', value=20.0, step=0.1)
        max_temp = st.number_input('Suhu Maksimum', value=30.0, step=0.1)

        if st.button('Simpan'):
            collection.insert_one({'min_temp': min_temp, 'max_temp': max_temp})
            st.success('Pengaturan suhu berhasil disimpan!')

        latest_setting = collection.find().sort([('_id', -1)]).limit(1)
        for setting in latest_setting:
            st.write('Suhu Minimum Terbaru:', setting['min_temp'])
            st.write('Suhu Maksimum Terbaru:', setting['max_temp'])

    elif menu_selection == "Object Detection":
        st.markdown("<h1 style='text-align: center;'>Object Detection for Sawi Varieties</h1>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True) 
        st.markdown("<h3 style='text-align: center;'>📷 Upload Image and Video</h3>", unsafe_allow_html=True)

        model = YOLO('model.pt')

        with st.form("object_detection_form"):
            st.write("Upload an image or video to detect objects using YOLOv8.")
            uploaded_file = st.file_uploader("Upload image or video", type=['jpg', 'png', 'mp4'], label_visibility='collapsed')
            min_confidence = st.slider('Confidence Score', 0.0, 1.0, 0.3)
            submit_button = st.form_submit_button(label='Submit')

        if uploaded_file is not None and submit_button:
            if uploaded_file.type.startswith('image'):
                detect_objects_in_image(model, uploaded_file, min_confidence)
            elif uploaded_file.type.startswith('video'):
                detect_objects_in_video(model, uploaded_file, min_confidence)

        st.write("")
        st.write("")
        st.markdown("<h3 style='text-align: center;'>📡 Real-time Object Detection from ESP32 CAM</h3>", unsafe_allow_html=True)

        # Create a form for webcam controls
        with st.form("webcam_form"):
            min_confidence = st.slider('Confidence Score', 0.0, 1.0, 0.3)
            submit_button_start = st.form_submit_button('Start Video Stream')
            submit_button_stop = st.form_submit_button('Stop Video Stream')

        if submit_button_start:
            # Jalankan deteksi objek secara real-time
            while True:
                img_resp = urllib.request.urlopen(url)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(imgnp, -1)
                result_frame = process_frame(frame, model, min_confidence)
                frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        if submit_button_stop:
            st.write("Video stream stopped.")

# Jalankan aplikasi Streamlit
if __name__ == '__main__':
    main()
