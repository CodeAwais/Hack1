from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv() #load the environment variables

router = APIRouter()

class DetailsRequest(BaseModel):
    label : str
    confidence : float
    age : int
    gender : str
    symptoms : list[str] = []


@router.post("/")
def generate_report(request: DetailsRequest):
    try:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Perplexity API Key is not set or invalid")


        symptoms_str = ""
        if request.symptoms:
            symptoms_str = " The patient has the following symptoms " + ", ".join(request.symptoms) + "."
        

        prompt = (
            f"Generate a detailed report for the patient with the following details: "
            f"The age of the patient is {request.age} and the gender is {request.gender}."
            f"The MRI scan suggests {request.label} with {request.confidence}% for confidence."
            f"and the symptoms the patient reported are {symptoms_str} \n\n"
            f"Add the following sections to the report: "
            f"1. Patient Summary\n"
            f"2. Risk Assessment\n"
            f"3. Explanation of the findings in layman's terms\n"
            f"4. Key MRI features that support the diagnosis\n"
            f"5. A visual 'Severity Risk Meter' using emojis and a pointer based on the confidence score\n"
            f"Be medically accurate and use plain language."

        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model" : "sonar", # add the correct model name here
            "messages" : [
                {
                    "role" : "user",
                    "content" : prompt
                }
            ]
        }

        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"API request failed! {response.text}")

        
        data = response.json()
        content = data.get("choices")[0].get("message").get("content")

        return {"summary" : content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))