from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import prediction, perplexity, generate_report

app = FastAPI(
    title="Multiple Sclerosis Detector",
    description="Detecting Multiple Sclerosis using MRI images and giving feedback using Perplexity Api",
) 

origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, #allow all origins
    allow_credentials=True,
    allow_methods=["*"], #allow all methods
    allow_headers=["*"], #allow all headers
)

app.include_router(prediction.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(perplexity.router, prefix="/api/perplexity", tags=["Perplexity Insights"])
app.include_router(generate_report.router, prefix="/api/generate_report", tags=["Perplexity Insights"])

@app.get("/")
async def root():
    return {"message": "Multiple Sclerosis Detector is running!"}







