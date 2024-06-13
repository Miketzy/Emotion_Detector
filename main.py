import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import base64
from tensorflow.keras.preprocessing.image import img_to_array

# Set up Streamlit page configuration
file_path = "image/logo.png"  # Replace with your image file path
background_image_path = "image/Main_BG.jpg"  # Replace with your background image file path

with open(file_path, "rb") as f:
    img_bytes = f.read()
encoded_img = base64.b64encode(img_bytes).decode()

st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon=f"data:image/png;base64,{encoded_img}",
    layout="centered",
)

# Read and encode background image
with open(background_image_path, "rb") as f:
    bg_img_bytes = f.read()
encoded_bg_img = base64.b64encode(bg_img_bytes).decode()

# Inject custom CSS for background image and other styles
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap');
    
    body {{
        background-image: url("data:image/png;base64,{encoded_bg_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    * {{
        font-family: "Poppins", sans-serif;
    }}
    .stButton>button {{
        background-color: #02A367;
        border: 2px solid #02A367;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: 700;
        margin: 3px 2px;
        cursor: pointer;
        border-radius: 10px;
        transition: .3s ease-in-out; 
    }}
    .stButton>button:hover {{
        border: 2px solid #02A367;
        background-color: #fff;
        color: #02A367;
    }}
    .stButton>button:active {{
        background-color: #02A367;
        border: 2px solid #02A367;
        color: white;
    }}
    .css-18e3th9 {{
        background-color: rgba(255, 255, 255, 0.8);
        padding-top: 2rem;
        padding-bottom: 10rem;
        padding-left: 2rem;
        padding-right: 2rem;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Load the TensorFlow model for emotion detection
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("model.h5")

emotion_model = load_emotion_model()

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion detection function
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        preds = emotion_model.predict(roi_gray)[0]
        label = emotion_labels[preds.argmax()]
        confidence = float(preds[preds.argmax()])

        results.append({
            "label": label,
            "confidence": confidence,
            "box": [x, y, w, h],
        })

    return results

# Streamlit app layout
st.header("Face Emotion Recognition")
st.write("Upload an image to detect the emotions on faces.")

uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", use_column_width=True)

    if st.button("Detect Emotion"):
        with st.spinner('Detecting...'):
            results = detect_emotion(image)
            st.write("Detected Emotions:")
            for result in results:
                label = result["label"]
                confidence = result["confidence"]
                st.write(f"{label}: {confidence * 100:.2f}%")
                x, y, w, h = result["box"]
                image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            st.image(image, channels="BGR", use_column_width=True)
