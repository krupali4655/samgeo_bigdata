

import os
import shutil
import tempfile
import time
import streamlit as st

from samgeo import split_raster
from samgeo.text_sam import LangSAM
import leafmap.foliumap as leafmap


def segment_image(image_file, text_prompt):
    try:
        start_time = time.time()
        temp_dir = tempfile.mkdtemp()

        # Save image
        image_path = os.path.join(temp_dir, "input_image.tif")
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        # Leafmap setup
        m = leafmap.Map()
        m.add_raster(image_path, layer_name="Input Image")

        # Tile image
        tile_dir = os.path.join(temp_dir, "tiles")
        split_raster(image_path, out_dir=tile_dir, tile_size=(1000, 1000), overlap=0)

        # LangSAM
        sam = LangSAM()

        mask_dir = os.path.join(temp_dir, "masks")
        sam.predict_batch(
            images=tile_dir,
            out_dir=mask_dir,
            text_prompt=text_prompt,
            box_threshold=0.24,
            text_threshold=0.24,
            mask_multiplier=255,
            dtype="uint8",
            merge=True,
        )

        output_tif = os.path.join(temp_dir, "output.tif")
        output_shp = os.path.join(temp_dir, "output.shp")

        sam.show_anns(
            cmap="Greys_r",
            add_boxes=False,
            alpha=1,
            title="Segmented Output",
            blend=False,
            output=output_tif,
        )

        sam.raster_to_vector(output_tif, output_shp)

        # Display map
        m.add_raster(output_tif, layer_name=text_prompt, palette="Greens", opacity=0.5, nodata=0)
        m.add_vector(output_shp, layer_name="Segmented Vector", style={
            "color": "#3388ff",
            "weight": 2,
            "fillColor": "#7c4185",
            "fillOpacity": 0.5,
        })

        st.success("Segmentation complete!")
        st.markdown("### Interactive Map")
        m.to_streamlit(height=600)

        st.download_button("Download Segmented TIF", open(output_tif, "rb"), file_name="output.tif")
        st.download_button("Download Shapefile (ZIP)", open(output_shp, "rb"), file_name="output.shp")

        elapsed = time.time() - start_time
        st.info(f"Inference Time: {elapsed:.2f} seconds")

    except Exception as e:
        st.error(f"Error: {str(e)}")


# Streamlit UI
st.title("Image Segmentation with LangSAM")
st.write("Upload a GeoTIFF image and enter a text prompt (e.g., 'Roads', 'Buildings').")

uploaded_image = st.file_uploader("Upload a TIFF Image", type=["tif", "tiff"])
text_prompt = st.text_input("Enter Text Prompt", "")

if uploaded_image and text_prompt:
    if st.button("Run Segmentation"):
        segment_image(uploaded_image, text_prompt)
