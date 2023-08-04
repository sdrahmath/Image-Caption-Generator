import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

@st.cache_data
def load_feature_extractor():
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return feature_extractor

def generate_caption(feature_extractor, image):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    i_image = Image.open(image)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def set_style():
    st.markdown(
        """
        <style>
        /* Use the path to the styles.css file relative to the app.py file */
        @import url("./styles.css");
        </style>
        """,
        unsafe_allow_html=True,
    )
def main():
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    set_style()
    feature_extractor = load_feature_extractor()

    st.title("Image Caption Generator")

    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image is not None:
        caption = generate_caption(feature_extractor, image)

        # Display the image with rounded borders
        st.image(
            image,
            caption="",
            use_column_width=True,
            clamp=True,
            channels="RGB",
        )

        # Display the caption in bold and large font size
        st.markdown(f"<p style='font-size: 24px; font-weight: bold; color:#000000;' > <b style= font-size: 50px; font-weight:bold; color: '#000000';'>CAPTION : </b> <b >{caption}.</b></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
