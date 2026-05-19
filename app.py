from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import os
import shutil
from Pipeline import pipeline, llm

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )
@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    await file.seek(0)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print(f"Processing {file_path}")
        prediction_data = pipeline(file_path)
        prediction_data["filename"] = file.filename
        return prediction_data
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post('/llm-explain')
async def get_llm_explanation(prediction_data: dict):
    def process_stream():
        stream = llm(prediction_data)
        for chunk in stream:
            yield chunk.message.content
    return StreamingResponse(process_stream(), media_type="text/plain")

DEFAULT_FILE = '/media/gowtham/MULTIMEDIA/Projects/ai-financial-risk-engine/AI-Financial-Risk-Engine/ais-doc/AIS_High_Risk_15_Pages.pdf'

@app.get('/predict')
def get_model_pred():
    prediction_data = pipeline(DEFAULT_FILE)
    return prediction_data

@app.get('/llm')
def get_llm_pred():
    def process_stream():
        prediction_data = pipeline(DEFAULT_FILE)
        stream = llm(prediction_data)
        for chunk in stream:
            yield chunk.message.content
    return StreamingResponse(process_stream(), media_type="text/plain")