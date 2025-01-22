import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# TensorFlow model prediction function
def model_prediction(image):
    model = tf.keras.models.load_model('pruned_model.keras')
    img = cv2.resize(image, (224,224))  # Resize image to match model input
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar configuration for Streamlit app
st.sidebar.title('Mobile Screen Inspection')
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "About", "Mobile Inspection", "Live Inspection"])

# Home page
if app_mode == "Home":
    st.header('MOBILE SCREEN INSPECTION SYSTEM')
    image_path = 'home_page.png'
    st.image(image_path, use_column_width=True)
    
    st.markdown('''
                Welcome to the Mobile Screen Inspection System.
                This system is designed to help you inspect your mobile screen whether it is OK or NG.
                Please select the "Mobile Inspection" mode to upload an image of the mobile screen.
                ''')

# About page
elif app_mode == "About":
    st.header("About")
    st.markdown('''
                ### About Dataset
                Go to Mobile Inspection mode to upload an image of the mobile screen. 
                The system will predict if there is a fault in the mobile screen.
                ''')

# Mobile Inspection page
elif app_mode == "Mobile Inspection":
    st.header('Mobile Inspection')
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, width=400, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            # Get prediction result
            result_index = model_prediction(np.array(image))
            
            # Reading Labels
            class_names = ['Good_Cam1','NG_Dust_Cam1']
            prediction = class_names[result_index]
            
            # Display the prediction
            st.write("Our Prediction: " + prediction)
            
            # Change background color based on prediction
            if prediction.startswith('Good'):
                st.markdown(
                    """
                    <style>
                    .reportview-container {
                        background-color: #DFFFD6; /* Light Green */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.success("Mobile Screen is: " + prediction)
            elif prediction.startswith('NG'):
                st.markdown(
                    """
                    <style>
                    .reportview-container {
                        background-color: #FFCCCB; /* Light Red */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.error("Mobile Screen is: " + prediction)

elif app_mode == "Live Inspection":
    st.header('Live Mobile Screen Inspection')
    
    # Start video capture using OpenCV
    run = st.checkbox('Run')
    
    FRAME_WINDOW = st.image([])  # Create an empty image placeholder
    
    cap = cv2.VideoCapture(1)  # Open default camera
    
    while run:
        ret, frame = cap.read()  # Read a frame from the camera
        
        if ret:
            # Make predictions on the current frame
            result_index = model_prediction(frame)
            
            # Reading Labels
            class_names = ['Good_Cam1','NG_Dust_Cam1']
            prediction = class_names[result_index]
            
            # Overlay prediction text on frame
            if prediction.startswith('Good'):
                color = (0, 255, 0)  # Green for Good
            elif prediction.startswith('NG'):
                color = (0, 0, 255)  # Red for NG (Scratch)
            
            cv2.putText(frame, f'Prediction: {prediction}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Convert BGR image (OpenCV format) to RGB format (Streamlit format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the resulting frame in Streamlit app
            FRAME_WINDOW.image(frame_rgb, channels='RGB')
    
    cap.release()  # Release the camera when done



