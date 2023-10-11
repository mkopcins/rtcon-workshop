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
    proba: List[float]
    labels: List[str]

# load model here
model = ...


@app.post("/api/mnist", response_model=PredResponse)
async def mnist_predict(img_data: ImageData):
    img = img_data.to_pil()
    img = img.resize((28, 28))

    img = to_tensor(img).unsqueeze(0)
    img = 1-img
    img = img * 2. - 1.

    output = model(img)
    probabilities = F.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1).item()

    return {
        "prediction": prediction,
        "proba": probabilities.tolist()[0],
        "labels": [str(i) for i in range(10)]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8081, log_level="info", reload=True)
