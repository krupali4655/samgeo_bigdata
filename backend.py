import os
import time
import torch
import psutil
import rasterio
from rasterio.crs import CRS
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from samgeo.text_sam import LangSAM
from samgeo import split_raster

app = FastAPI()

# Define paths
BASE_DIR = "D:/KRUPALI/test"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TILES_DIR = os.path.join(BASE_DIR, "tiles")
MASKS_DIR = os.path.join(BASE_DIR, "masks")

# Ensure directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

def get_system_resources():
    """Fetch CPU, RAM, and GPU usage statistics."""
    gpu_usage = "N/A"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
        gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
        gpu_usage = f"{gpu_name} | Memory: {gpu_mem_alloc:.2f} GB / {gpu_mem_total:.2f} GB"

    return {
        "CPU_Usage": f"{psutil.cpu_percent()}%",
        "RAM_Usage": f"{psutil.virtual_memory().percent}%",
        "GPU_Usage": gpu_usage,
    }

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...), text_prompt: str = Form(...)):
    """Processes the uploaded image and performs segmentation."""
    try:
        start_time = time.time()
        system_before = get_system_resources()

        # Save uploaded image
        image_path = os.path.join(BASE_DIR, "input_image.tif")
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Initialize LangSAM
        sam = LangSAM()

        # Split image into tiles
        split_raster(image_path, out_dir=TILES_DIR, tile_size=(512, 512), overlap=0)

        # Perform segmentation
        sam.predict_batch(
            images=TILES_DIR,
            out_dir=MASKS_DIR,
            text_prompt=text_prompt,
            box_threshold=0.24,
            text_threshold=0.24,
            mask_multiplier=255,
            dtype="uint8",
            merge=True,
        )

        # Define final output paths
        output_tif = os.path.join(OUTPUTS_DIR, f"segmented_{int(time.time())}.tif")

        # Move the final mask file to output directory
        merged_tif = os.path.join(MASKS_DIR, "merged.tif")
        if os.path.exists(merged_tif):
            os.rename(merged_tif, output_tif)
        else:
            raise ValueError("Merged TIFF file not found!")

        # Ensure the output file exists
        if not os.path.exists(output_tif):
            raise ValueError(f"Output TIFF file not found: {output_tif}")

        if os.stat(output_tif).st_size == 0:
            raise ValueError("Output TIFF file is empty!")

        # Extract CRS from raster file
        with rasterio.open(output_tif) as src:
            crs_info = str(src.crs)  # Convert CRS object to string

        # Compute processing time
        inference_time = time.time() - start_time
        system_after = get_system_resources()

        return JSONResponse(
            content={
                "message": "Segmentation Completed",
                "inference_time": inference_time,
                "system_before": system_before,
                "system_after": system_after,
                "output_tif": f"/download/tif/{os.path.basename(output_tif)}",
                "crs_info": crs_info  # Now properly defined
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/download/tif/{filename}")
async def download_tif(filename: str):
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/tiff", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

















































#WORKING SMOOTH LIKE BUTTER , DONT TOUCH IT!!!!!!!!!!!!!!!!!!!!!!!!!!
# import os
# import time
# import torch
# import psutil
# import rasterio
# from rasterio.crs import CRS
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.responses import FileResponse, JSONResponse
# from samgeo.text_sam import LangSAM
# from samgeo import split_raster

# app = FastAPI()

# # Define paths
# BASE_DIR = "D:/KRUPALI/test"
# OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
# TILES_DIR = os.path.join(BASE_DIR, "tiles")
# MASKS_DIR = os.path.join(BASE_DIR, "masks")

# # Ensure directories exist
# os.makedirs(OUTPUTS_DIR, exist_ok=True)
# os.makedirs(TILES_DIR, exist_ok=True)
# os.makedirs(MASKS_DIR, exist_ok=True)

# def get_system_resources():
#     """Fetch CPU, RAM, and GPU usage statistics."""
#     gpu_usage = "N/A"
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         gpu_name = torch.cuda.get_device_name(device)
#         gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
#         gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
#         gpu_usage = f"{gpu_name} | Memory: {gpu_mem_alloc:.2f} GB / {gpu_mem_total:.2f} GB"

#     return {
#         "CPU_Usage": f"{psutil.cpu_percent()}%",
#         "RAM_Usage": f"{psutil.virtual_memory().percent}%",
#         "GPU_Usage": gpu_usage,
#     }

# @app.post("/segment/")
# async def segment_image(file: UploadFile = File(...), text_prompt: str = Form(...)):
#     """Processes the uploaded image and performs segmentation."""
#     try:
#         start_time = time.time()
#         system_before = get_system_resources()

#         # Save uploaded image
#         image_path = os.path.join(BASE_DIR, "input_image.tif")
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Initialize LangSAM
#         sam = LangSAM()

#         # Split image into tiles
#         split_raster(image_path, out_dir=TILES_DIR, tile_size=(512, 512), overlap=0)

#         # Perform segmentation
#         sam.predict_batch(
#             images=TILES_DIR,
#             out_dir=MASKS_DIR,
#             text_prompt=text_prompt,
#             box_threshold=0.24,
#             text_threshold=0.24,
#             mask_multiplier=255,
#             dtype="uint8",
#             merge=True,
#         )

#         # Define final output paths
#         output_tif = os.path.join(OUTPUTS_DIR, f"segmented_{int(time.time())}.tif")

#         # Move the final mask file to output directory
#         merged_tif = os.path.join(MASKS_DIR, "merged.tif")
#         if os.path.exists(merged_tif):
#             os.rename(merged_tif, output_tif)
#         else:
#             raise ValueError("Merged TIFF file not found!")

#         # Ensure the output file exists
#         if not os.path.exists(output_tif):
#             raise ValueError(f"Output TIFF file not found: {output_tif}")

#         if os.stat(output_tif).st_size == 0:
#             raise ValueError("Output TIFF file is empty!")

#         # Extract CRS from raster file
#         with rasterio.open(output_tif) as src:
#             crs_info = str(src.crs)  # Convert CRS object to string

#         # Compute processing time
#         inference_time = time.time() - start_time
#         system_after = get_system_resources()

#         return JSONResponse(
#             content={
#                 "message": "Segmentation Completed",
#                 "inference_time": inference_time,
#                 "system_before": system_before,
#                 "system_after": system_after,
#                 "output_tif": f"/download/tif/{os.path.basename(output_tif)}",
#                 "crs_info": crs_info  # Now properly defined
#             }
#         )

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.get("/download/tif/{filename}")
# async def download_tif(filename: str):
#     file_path = os.path.join(OUTPUTS_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path, media_type="image/tiff", filename=filename)
#     raise HTTPException(status_code=404, detail="File not found")
























# #till 2nd progress day

# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, FileResponse
# import os
# import numpy as np
# import traceback  
# from samgeo import split_raster
# from samgeo.text_sam import LangSAM
# from PIL import Image
# import leafmap
# import shutil

# app = FastAPI()

# # Ensure directories exist
# os.makedirs("tiles", exist_ok=True)
# os.makedirs("masks", exist_ok=True)
# os.makedirs("uploads", exist_ok=True)
# os.makedirs("vectors", exist_ok=True)

# sam = LangSAM()

# @app.post("/segment/")
# async def segment_image(
#     image: UploadFile = File(...),  
#     text_prompt: str = Form(...)
# ):
#     try:
#         # Save uploaded image
#         image_path = f"uploads/{image.filename}"
#         with open(image_path, "wb") as f:
#             f.write(await image.read())

#         # Remove old mask if it exists
#         mask_path = "masks/merged.tif"
#         if os.path.exists(mask_path):
#             try:
#                 os.remove(mask_path)
#             except Exception as e:
#                 return JSONResponse(content={"status": "error", "message": f"Failed to delete old mask: {str(e)}"}, status_code=500)

#         # Split image into tiles
#         split_raster(image_path, out_dir="tiles", tile_size=(1000, 1000), overlap=0)

#         # Run segmentation
#         masks = sam.predict_batch(
#             images="tiles",
#             out_dir="masks",
#             text_prompt=text_prompt,
#             box_threshold=0.24,
#             text_threshold=0.24,
#             mask_multiplier=255,
#             dtype="uint8",
#             merge=True,
#             verbose=True,
#         )

#         # Check if segmentation output is valid
#         if masks is None:
#             return JSONResponse(content={"status": "error", "message": "Segmentation returned None."}, status_code=500)

#         mask_array = np.array(masks)

#         if mask_array.size == 0:
#             return JSONResponse(content={"status": "error", "message": "Segmentation mask is empty."}, status_code=500)

#         # Convert dtype to uint8 before saving
#         mask_array = mask_array.astype(np.uint8)

#         img = Image.fromarray(mask_array.squeeze())
#         img.save(mask_path)

#         # Visualization with bounding boxes
#         sam.show_anns(
#             cmap="Greens",
#             box_color="red",
#             title="Automatic Segmentation of Trees",
#             blend=True,
#         )

#         # Visualization without bounding boxes
#         sam.show_anns(
#             cmap="Greens",
#             add_boxes=False,
#             alpha=0.5,
#             title="Automatic Segmentation of Trees",
#         )

#         # Visualization grayscale image
#         trees_tif = "masks/trees.tif"
#         sam.show_anns(
#             cmap="Greys_r",
#             add_boxes=False,
#             alpha=1,
#             title="Automatic Segmentation of Trees",
#             blend=False,
#             output=trees_tif,
#         )

#         # Convert the result to vector format
#         vector_path = "vectors/trees.shp"
#         sam.raster_to_vector(trees_tif, vector_path)

#         return JSONResponse(content={"status": "success", "output": mask_path, "vector": vector_path})

#     except Exception as e:
#         error_details = traceback.format_exc()  
#         print(" ERROR: An exception occurred:\n", error_details)  

#         return JSONResponse(content={"status": "error", "message": str(e), "details": error_details}, status_code=500)




