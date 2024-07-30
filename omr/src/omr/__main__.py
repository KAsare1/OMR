import os
import uuid
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import *






app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

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