import streamlit as st
from PIL import Image
import numpy as np
import os
import shutil
from datetime import date
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from openai import OpenAI

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# (We are mocking these for initial deployment test)
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# import openai

# ==========================================================
# 1. CONFIGURATION & DESIGN
# ==========================================================
st.set_page_config(
    page_title="FloraMind: AI Botanical Diagnostic",
    page_icon="🌸",
    layout="wide", # Essential for desktop responsiveness
    initial_sidebar_state="expanded"
)

# Set up the visual palette (Botanical Greens and Medical Blues)
PRIMARY_COLOR = "#2B65EC" # Professional Medical Blue
SECONDARY_COLOR = "#1A531A" # Deep Botanical Green
TEXT_COLOR = "#333333"

# Inject custom CSS for mobile responsiveness and professional styling
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: #f8f9fa;
    }}
    
    /* Center the title and logo */
    .title-text {{
        color: {SECONDARY_COLOR};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-align: center;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }}
    
    .subtitle-text {{
        text-align: center;
        color: {TEXT_COLOR};
        font-weight: 400;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }}

    /* Result Card Styling */
    .result-card {{
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
        border-left: 5px solid {PRIMARY_COLOR};
    }}

    /* Section Headers */
    .section-header {{
        color: {PRIMARY_COLOR};
        font-weight: 600;
        font-size: 1.4rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    /* Mobile responsiveness tweaks */
    @media (max-width: 768px) {{
        .title-text {{ font-size: 2rem; }}
        .result-card {{ padding: 1rem; }}
        /* Make columns stack vertically on mobile */
        [data-testid="column"] {{
            width: 100% !important;
            margin-bottom: 1rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. APP STATE & MOCKING (Day 8-9 placeholders)
# ==========================================================

# Placeholders for the model you will load later (Day 10)
CLASS_NAMES = ['dandelion', 'iris', 'lotus', 'rose', 'sunflower', 'tulip']

@st.cache_resource
# def load_classification_model():
#     """ Day 10 task: Load your saved .keras model here. """
#     st.info(" Day 8-10 Placeholder: Model loading will be active soon.")
#     # return tf.keras.models.load_model('Flower6_VGG16_Final.keras')
#     return None
def load_classification_model():
    model_path = 'Flower6_VGG16_Final.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found!")
        return None
# def mock_predict_image(pil_image):
#     """ Day 10 task: Connect VGG16 prediction logic. Currently mocked. """
#     # Mimic the high confidence from your training matrix
#     return 'rose', 98.6

def predict_image(pil_image, model):
    # Match the input size used in your training (likely 224x224 for VGG16)
    img = pil_image.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess using VGG16 specific normalization
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    score_index = np.argmax(predictions[0])
    
    confidence = predictions[0][score_index] * 100
    predicted_class = CLASS_NAMES[score_index]
    
    return predicted_class, confidence

# Day 11 placeholder for OpenAI
# api_key = st.secrets["OPENAI_API_KEY"] # Set this in streamlit secrets later
# openai.api_key = api_key

# def mock_get_medicinal_info(flower_name):
#     """ Day 9-11 task: Activate OpenAI integration. Currently mocked. """
#     # MOCK DATA Structure (to match Day 9 objective)
#     mock_data = {
#         'rose': """The rose is prized not just for its scent. Traditionally, rose hips (the fruit) are exceptionally high in Vitamin C, often used to make teas that may aid immune function and reduce inflammation. Essential oil from the petals is used aromatically to ease stress. **WARNING: Always consult a professional before medicinal use.**""",
#         'dandelion': """Considered a weed by many, dandelions are packed with nutrients. Historically used as a diuretic, they may support kidney health. The root is often utilized as a digestive aid. Dandelion greens can be eaten and are high in vitamins A, C, and K."""
#     }
#     return mock_data.get(flower_name, "Information regarding the traditional use of this specific flower is currently unavailable. Medicinal use requires specialist knowledge and consultation with a healthcare provider.")


# def get_medicinal_info(flower_name):
#     # Note: Ensure you set your API key in .streamlit/secrets.toml
#     # or use: client = openai.OpenAI(api_key="your_key_here")
    
#     prompt = f"""
#         You are a botanical and medical expert.

#         Give structured information about the {flower_name} flower:
#         - Key medicinal uses
#         - Active compounds (if known)
#         - Traditional uses
#         - Any scientific evidence
#         - Safety warning

#         Keep it concise and professional.
#         """
    
#     # try:
#     #     # This uses the modern OpenAI SDK (v1.0+)
#     #     client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
#     #     response = client.chat.completions.create(
#     #         model="gpt-3.5-turbo",
#     #         messages=[{"role": "user", "content": prompt}]
#     #     )
#     #     return response.choices[0].message.content
#     from openai import OpenAI  # ADD at top of file

def get_medicinal_info(flower_name):
    # prompt = f"""
    # Provide a professional and concise summary of the medicinal benefits 
    # and traditional uses of the {flower_name} flower.
    # Also include a safety disclaimer.
    # """
    prompt = f"""
        You are a botanical and medical expert.

        Give structured information about the {flower_name} flower:
        - Key medicinal uses
        - Active compounds (if known)
        - Traditional uses
        - Any scientific evidence
        - Safety warning

        Keep it concise and professional.
        """

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.secrets["OPENROUTER_API_KEY"]  # store key here
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"reasoning": {"enabled": True}}
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error fetching medicinal info: {str(e)}"
    # except Exception as e:
    #     # Fallback to your local mock data if API fails or key is missing
    #     return f"Could not connect to AI Research module. (Error: {str(e)})"

# ==========================================================
# 3. SIDEBAR / NAV
# ==========================================================
with st.sidebar:
    # Display the professional logo
    try:
        logo = Image.open('FloraMind.png')
        st.image(logo, width='stretch')
    except FileNotFoundError:
        st.error("Logo file 'FloraMind.png' not found. Ensure it is in the same directory.")
        
    st.markdown("---")
    st.markdown(f"**FloraMind** AI Botanical Diagnostic © {date.today().year}")
    st.markdown("A sophisticated classification engine for identifying flora and exploring traditional therapeutic applications.")
    
    st.markdown("---")
    st.markdown("### How to Use:")
    st.write("1. Choose input: Upload a stored image or use your device's Camera.")
    st.write("2. Ensure the flower is clearly visible and in focus.")
    st.write("3. View classification and verified botanical summaries.")

# ==========================================================
# 4. MAIN PAGE CONTENT
# ==========================================================
st.markdown("<h1 class='title-text'>🌸 FloraMind</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>AI Botanical Classification & Diagnostic Summary</p>", unsafe_allow_html=True)

# Main UI Tabs: Select Input Source (Crucial for Mobile/Desktop flexibility)
tab_desktop, tab_mobile = st.tabs(["[💻] Upload Stored Image", "[📷] Use Device Camera"])

uploaded_file = None
camera_photo = None

# Tab 1: For Desktop users with files
with tab_desktop:
    st.subheader("Classification Queue")
    uploaded_file = st.file_uploader("Select a botanical specimen image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

# Tab 2: For Mobile users or Desktop with webcam
# with tab_mobile:
#     st.subheader("Live Acquisition")
#     camera_photo = st.camera_input("Acquire Image from Camera")

# Tab 2: For Mobile users or Desktop with webcam
with tab_mobile:
    st.subheader("Live Acquisition")
    
    # Use a session state or a button to trigger the camera
    if st.button("Activate Camera"):
        st.session_state.camera_on = True

    if st.session_state.get("camera_on", False):
        camera_photo = st.camera_input("Acquire Image from Camera")
        
        # Optional: Add a button to turn it back off
        if st.button("Close Camera"):
            st.session_state.camera_on = False
            st.rerun()

# Consolidate input from either source
final_image_input = None
if uploaded_file is not None:
    final_image_input = Image.open(uploaded_file)
elif camera_photo is not None:
    final_image_input = Image.open(camera_photo)

# ==========================================================
# 5. PREDICTION & RESULTS (The core workflow)
# ==========================================================
model = load_classification_model()
if final_image_input is not None:
    # Set up the results area in columns: 1 (Image) and 2 (Analysis)
    col_img, col_analysis = st.columns([1, 1.5]) # Desktop responsive layout
    
    with col_img:
        st.image(final_image_input, caption='Acquired Specimen', width='stretch')

    with col_analysis:
        st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-header'>Analysis Report</h3>", unsafe_allow_html=True)

        # 1. Run Classification (with Day 12 Polish: Loading Spinner)
        with st.spinner("Executing convolutional neural network classification..."):
            pred_class, confidence = predict_image(final_image_input, model)
        
        st.markdown(f"**Predicted Genus/Species:**")
        st.success(f"**{pred_class.upper()}** (Classification Score: {confidence:.1f}%)")

        # 2. Run Medicinal (Day 9 OpenAI integration)
        st.markdown("<h3 class='section-header'>Traditional/Botanical Significance</h3>", unsafe_allow_html=True)
        
        # In a real sprint, this info call might be slow, so we use another spinner
        with st.spinner(f"Retrieving diagnostic summary for {pred_class}..."):
            medicinal_summary = get_medicinal_info(pred_class)
            
        st.info(medicinal_summary)
        
        # Day 12 Polish: Final professional footer inside the card
        st.markdown("---")
        st.caption(f"FloraMind Diagnostic Engine v1.0 | Analysis timestamp: {date.today()}")
        st.markdown(f"</div>", unsafe_allow_html=True)

else:
    # Welcome / State message when no image is selected
    st.warning("👈 Please select an image acquisition method to begin the diagnostic workflow.")

# ==========================================================
# 6. HOW TO RUN
# ==========================================================
# 1. Save this code as `floramine_app.py`
# 2. Save the logo as `logo.png` in the same folder.
# 3. Install requirements: pip install streamlit pillow numpy (later: tensorflow openai)
# 4. Run: streamlit run floramine_app.py