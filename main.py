import base64
import io
from typing import List

import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from torchvision.transforms.functional import to_tensor
import torch


from model import Net, CNN

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageData(BaseModel):
    img_base_64: str

    def to_pil(self):
        decoded_image = base64.b64decode(self.img_base_64)
        img = Image.open(io.BytesIO(decoded_image))
        img = img.convert('L')
        return img


class PredResponse(BaseModel):
    prediction: int


import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession('/Users/kopcion/swm-ai/rtcon/resnet101.onnx')



@app.post("/api/mnist", response_model=PredResponse)
async def mnist_predict(img_data: ImageData):
    img = img_data.to_pil()
    img = img.resize((224, 224))

    img = to_tensor(img).unsqueeze(0)
    output = torch.from_numpy(ort_sess.run(None, {'input': img.numpy()}))

    prediction = torch.argmax(output, dim=1).item()

    return {
        "prediction": prediction,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8081, log_level="info", reload=True)
