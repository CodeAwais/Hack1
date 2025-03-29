from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

import io 

router = APIRouter()

@router.post("/")
async def predict_ms(file : UploadFile = File(...)):

    try:
        if file.content_type != "image/jpeg" and file.content_type != "image/png" and file.content_type != "image/dcm":
            raise HTTPException(status_code=400, detail="File must be a JPEG, PNG or DICOM Image")
        
        image_data = await file.read()

        # open and process the image here 
        
        #Use the model to predict the disease 
        
        # result here 

        # return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
