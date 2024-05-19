from typing import Literal, Optional
from omegaconf import OmegaConf
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image


class Application:
    def __init__(self, config: OmegaConf) -> None:
        self.config = OmegaConf.to_container(cfg=config, resolve=True)
        self.setup_streamlit()

    def setup_streamlit(self) -> None:
        st.title(body="Peril Scan")
        self.uploaded_file: Optional[UploadedFile] = st.file_uploader(
            label="Choose an image...", type="jpg"
        )

    def run(self) -> None:
        if self.uploaded_file is not None:
            image: Image.Image = Image.open(self.uploaded_file)
            st.image(image=image, caption="Uploaded Image.", use_column_width=True)

            # Generate caption
            caption: Literal["Caption"] = self.generate_caption(image=image)
            st.write("Generated Caption:", caption)

            # Classify image
            hazard: Literal["Classification"] = self.classify_image(image=image)
            st.write("Category:", hazard)

    def generate_caption(self, image: Image.Image) -> Literal["Caption"]:
        return "Caption"

    def classify_image(self, image: Image.Image) -> Literal["Classification"]:
        return "Classification"
