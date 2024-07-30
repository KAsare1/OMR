import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import marking, compare_grades, marker  # Ensure these functions are correctly imported

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def save_file_to_server(file: UploadFile):
    file_extension = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     file_path = save_file_to_server(file)
#     marks = marker(file_path)
#     return {"results": marks}

# @app.post("/compare/")
# async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     file_path1 = save_file_to_server(file1)
#     file_path2 = save_file_to_server(file2)

#     dict1 = dict_image(file_path1)
#     dict2 = dict_image(file_path2)

#     marked_dict1 = {k: marking(v) for k, v in dict1.items()}
#     marked_dict2 = {k: marking(v) for k, v in dict2.items()}

#     result = compare_grades(marked_dict1, marked_dict2)
#     return {"total_grade": result}

@app.post("/mark_picture_scheme")
async def score_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    file_path1 = save_file_to_server(file1)
    file_path2 = save_file_to_server(file2)

    marks = marker(file_path1)
    marking_scheme = marker(file_path2)

    results = compare_grades(marks, marking_scheme)
    return {"total score": results}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
