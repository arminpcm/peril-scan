from typing import List, Optional
from omegaconf import OmegaConf
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image
import logging

from lib.classify.clip_inference import ClipInference
from lib.caption.llava_inference import LlavaInference


class Application:
    def __init__(self, config: OmegaConf) -> None:
        self.config: OmegaConf = config
        config_dict = OmegaConf.to_container(cfg=config, resolve=True)
        logging.info(msg=config_dict)
        self.setup_streamlit()
        self.setup_application()

    def setup_application(self) -> None:
        self.classifier = ClipInference(config=self.config.classification)
        self.describer = LlavaInference(config=self.config.description)

    def setup_streamlit(self) -> None:
        st.title(body="Peril Scan")
        self.uploaded_file: Optional[UploadedFile] = st.file_uploader(
            label="Choose an image...", type="jpg"
        )

    def run(self) -> None:
        if self.uploaded_file is not None:
            image: Image.Image = Image.open(fp=self.uploaded_file)
            st.image(image=image, caption="Uploaded Image.", use_column_width=True)

            # Generate caption
            caption: str = self.generate_caption(image=image)
            st.write("OSHA Violation Summary:", caption)

            # Classify image
            hazard: str = self.classify_image(image=image)
            st.write("Category:", hazard)

    def generate_caption(self, image: Image.Image) -> str:
        input_data_dict = {"Image": image, "Prompt": self.config.description.prompt}
        prediction = self.describer.predict(input_data=input_data_dict)
        return prediction

    def classify_image(self, image: Image.Image) -> str:
        labels: List[str] = [category for category in self.config.categories]
        input_data_dict = {"Image": image, "Labels": labels}
        prediction = self.classifier.predict(input_data=input_data_dict)
        index = prediction["class"]
        if index == -1:
            return "Unknown"
        return labels[index]
