from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"], 
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
poke_df = pd.read_csv('data/pokedex_info.csv').sort_values(by = "Name")

@app.on_event("startup")
def load_model():
    global model 
    model = models.resnet18(weights = None)
    model.fc = nn.Sequential(nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 150))
    model.load_state_dict(torch.load('models/pokemodel.pt', weights_only = True, map_location = torch.device('cpu')),)
    model.eval()


def preprocess_image(file: UploadFile):
    transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    img = Image.open(file.file).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    return input_tensor

def predict(tensor):
    with torch.no_grad():
        output = model(tensor)
        class_probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(class_probabilities, dim=1).item()
        pokemon = poke_df.iloc[predicted_class]
        return pokemon.to_dict()

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    tensor = preprocess_image(file)
    pred = predict(tensor)
    return {"prediction": pred}