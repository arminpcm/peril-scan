import streamlit as st
from PIL import Image
import yaml


# Helper functions for image captioning and classification
def generate_caption(image):
    return "Caption"

def classify_image(image):
    return "Classification"



# Run the app
if __name__ == '__main__':
    # Streamlit app
    st.title("Peril Scan")

    # Load configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Generate caption
        caption = generate_caption(image)
        st.write("Generated Caption:", caption)

        # Classify image
        hazard = classify_image(image)
        
        st.write("Category:", hazard)
