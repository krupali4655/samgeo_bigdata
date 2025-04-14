import streamlit as st
import requests
import os
import io
from PIL import Image

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

st.title("Satellite Image Segmentation with LangSAM")

# File uploader
uploaded_file = st.file_uploader("Upload a satellite image (TIF format)", type=["tif"])

# Text prompt for segmentation
text_prompt = st.text_input("Enter segmentation text prompt:", "Detect water bodies")

if uploaded_file:
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Run Segmentation"):
    if uploaded_file is None:
        st.error("Please upload an image first!")
    else:
        # Send file to backend
        files = {"file": uploaded_file.getvalue()}
        data = {"text_prompt": text_prompt}
        response = requests.post(f"{BASE_URL}/segment/", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            st.success(result["message"])

            # Display system resource usage
            st.subheader("System Resource Usage (Before & After)")
            st.json(result["system_before"])
            st.json(result["system_after"])

            # Show inference time
            st.subheader("Inference Time")
            st.write(f"{result['inference_time']:.2f} seconds")

            # Download segmented file
            output_tif_url = f"{BASE_URL}{result['output_tif']}"
            st.markdown(f"[Download Segmented TIFF]({output_tif_url})")

            # Display CRS Info
            st.subheader("Coordinate Reference System (CRS)")
            st.code(result["crs_info"])

            # Display segmented image
            segmented_response = requests.get(output_tif_url)
            if segmented_response.status_code == 200:
                segmented_image = Image.open(io.BytesIO(segmented_response.content))
                st.subheader("Segmented Output")
                st.image(segmented_image, caption="Segmented Image", use_column_width=True)
            else:
                st.error("Failed to load segmented image.")
        else:
            st.error(f"Segmentation failed: {response.json().get('error', 'Unknown error')}")



































# WORKING SMOOTH LIKE BUTTER , DONT TOUCH IT !!!!!!!!!!!!!!!!!!
# import streamlit as st
# import requests
# import os

# # Backend URL
# BASE_URL = "http://127.0.0.1:8000"

# st.title("Satellite Image Segmentation with LangSAM")

# # File uploader
# uploaded_file = st.file_uploader("Upload a satellite image (TIF format)", type=["tif"])

# # Text prompt for segmentation
# text_prompt = st.text_input("Enter segmentation text prompt:", "Detect water bodies")

# if st.button("Run Segmentation"):
#     if uploaded_file is None:
#         st.error("Please upload an image first!")
#     else:
#         # Send file to backend
#         files = {"file": uploaded_file.getvalue()}
#         data = {"text_prompt": text_prompt}
#         response = requests.post(f"{BASE_URL}/segment/", files=files, data=data)

#         if response.status_code == 200:
#             result = response.json()
#             st.success(result["message"])

#             # Display system resource usage
#             st.subheader("System Resource Usage (Before & After)")
#             st.json(result["system_before"])
#             st.json(result["system_after"])

#             # Download segmented file
#             output_tif_url = f"{BASE_URL}{result['output_tif']}"
#             st.markdown(f"[Download Segmented TIFF]({output_tif_url})")

            
#             # Display CRS Info
#             st.subheader("Coordinate Reference System (CRS)")
#             st.code(result["crs_info"])

#         else:
#             st.error(f"Segmentation failed: {response.json().get('error', 'Unknown error')}")


































# #til 2nd progress day

# import streamlit as st
# import requests
# import leafmap
# import os

# st.title("SAM-based Satellite Image Segmentation")

# uploaded_image = st.file_uploader("Upload a satellite image", type=["tiff", "png", "jpg"])

# text_prompt = st.text_input("Enter a text prompt for segmentation", "tree")

# backend_url = "http://127.0.0.1:8000/segment/"  

# if st.button("Segment Image"):
#     if uploaded_image:
#         # Save uploaded image
#         image_path = os.path.join("uploads", uploaded_image.name)
#         with open(image_path, "wb") as f:
#             f.write(uploaded_image.getvalue())

#         try:
#             # Send the request to the backend
#             with open(image_path, "rb") as file:
#                 files = {"image": file}
#                 data = {"text_prompt": text_prompt}
#                 response = requests.post(backend_url, files=files, data=data)

#             if response.status_code == 200:
#                 result = response.json()
#                 if result["status"] == "success":
#                     st.success("Segmentation completed!")

#                     # Display images only if they exist
#                     mask_path = "masks/merged.tif"
#                     grayscale_path = "masks/trees.tif"
#                     vector_path = result.get("vector")

#                     if os.path.exists(mask_path):
#                         st.image(mask_path, caption="Segmented Image (Merged Mask)", use_column_width=True)
#                     else:
#                         st.warning("Segmented mask not found.")

#                     if os.path.exists(grayscale_path):
#                         st.image(grayscale_path, caption="Grayscale Segmentation", use_column_width=True)
#                     else:
#                         st.warning("Grayscale segmentation not found.")

#                     # Provide download link for vector output
#                     if vector_path and os.path.exists(vector_path):
#                         with open(vector_path, "rb") as file:
#                             st.download_button("Download Vector (Shapefile)", file, file_name="trees.shp")
#                     else:
#                         st.warning("Vector output not available.")
#                 else:
#                     st.error(result["message"])
#             else:
#                 st.error("Segmentation failed. Check the backend logs.")

#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# # Leafmap visualization
# m = leafmap.Map(center=[-22.1278, -51.4430], zoom=17, height="500px")
# m.add_basemap("SATELLITE")

# if uploaded_image:
#     image_path = os.path.join("uploads", uploaded_image.name)

#     if os.path.exists(image_path):
#         m.add_raster(image_path, layer_name="Uploaded Image")

#     # Add segmented result only if files exist
#     if os.path.exists("masks/merged.tif"):
#         m.add_raster("masks/merged.tif", layer_name="Segmented Mask")

#     if os.path.exists("masks/trees.tif"):
#         m.add_raster("masks/trees.tif", layer_name="Grayscale Mask")

#     if os.path.exists("vectors/trees.shp"):
#         m.add_shapefile("vectors/trees.shp", layer_name="Vector Output")

# st.components.v1.html(m.to_html(), height=600)










