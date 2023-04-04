import csv, codecs

import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from domain.chatGPT import get_gpt_description
from models.requests import *

app = FastAPI()

# this will allow the browser to open the response from the server
# in production, this would be a hardcoded DNS
origins = [ 
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
def health_check():
    return {"data": 42}


@app.post("/upload")
async def post_upload(file: UploadFile):
    from io import StringIO
    df = pd.read_csv(file.file)

    recommendation = await get_gpt_description(df)
    return {"description": recommendation}


@app.post("/additional_results")
async def post_additional_results(payload: AdditionalResults):
    newDescription = payload.description + "More Recommendations"
    return {
        "description": newDescription
    }
