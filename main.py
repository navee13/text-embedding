from fastapi import FastAPI, HTTPException
from endpoints import text_embedded
import requests
import onnxruntime as ort
import numpy as np
import gzip
import os


app = FastAPI()


# Load ONNX model from cloud bucket

bucket_text = (
    "https://storage.cloud.google.com/embedding_model_1/clip_text_model_vitb32.onnx"
)


app.include_router(text_embedded.router, tags=["Text-embedded microservice"])
