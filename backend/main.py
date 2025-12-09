from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from ml_service import MLEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_engine = MLEngine()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "ML Pipeline Backend is Running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    metadata = ml_engine.load_data(file_location)
    return metadata

@app.post("/run")
async def run_pipeline(
    target_column: str = Form(...),
    split_ratio: float = Form(...),
    preprocessing: str = Form(...),
    model_type: str = Form(...)
):
    result = ml_engine.run_pipeline(
        target_column, split_ratio, preprocessing, model_type
    )
    return result
