from fastapi import FastAPI,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
from keras.models import load_model
from app.code import predictcar
from httpx import ReadTimeout
import numpy as np
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)


PATHPERPROCESING = 'http://172.17.0.2:80/tovector'

PATHMODELFULL = r'./model/my_model.h5'

model = load_model(PATHMODELFULL)


@app.get("/")
def root():
    return {"message": "This is my api imageCAR"}    


@app.post("/Container2Perdiction/upload")
async def upload_file(audio_file: UploadFile):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(PATHPERPROCESING, files={"audio_file": (audio_file.filename, audio_file.file)})
            arrsound = response.json()
            nparry = np.array(arrsound['vecsound'])
            res = predictcar(model, nparry)
            print(res)
            return {"predict": res}
    except ReadTimeout:
        return {"error": "The request to the processing server timed out."}
    except Exception as e:
        return {"error": str(e)}
    

