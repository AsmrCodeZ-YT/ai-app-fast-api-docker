from typing import Union
from fastapi import FastAPI, UploadFile
from vilt_model import answer_pipline
from PIL import Image
import io

app = FastAPI()


@app.get("/")
async def read_root():
    return {"WelCome": "Im TheCodeZ!"}


@app.post("/predic/")
async def read_item(text: str, image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))

    result = answer_pipline(text, image)
    return {"result": result}
