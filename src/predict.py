import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image


def predict_page():

    #LOAD TRAINED MODEL
    model = models.resnet101(weights="ResNet101_Weights.DEFAULT")
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Linear(512, 150))
    model.load_state_dict(torch.load('../models/pokemodel.pt', weights_only = True, map_location = torch.device('cpu')),)
    model.eval()
    

    #DATA TRANSFORMATIONS
    transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


    # PROMPT USER IMAGE UPLOAD
    img = st.file_uploader('Upload image of Pokemon: ', type = ['jpeg', 'jpg', 'png'])
    poke_df = pd.read_csv('../data/pokedex_info.csv').sort_values(by = "Name")

    if img is not None:
        img = Image.open(img).convert('RGB')
        input_tensor = transform(img).unsqueeze(0)

        # PERFORM INFERENCE
        with torch.no_grad():
            output = model(input_tensor)

        # COMPUTING PREDICTED POKEMON
        class_probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(class_probabilities, dim=1).item()

        pokemon = poke_df.iloc[predicted_class]

        # UPLOAD POKEMON INFORMATION
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f'https://img.pokemondb.net/sprites/black-white/anim/normal/{pokemon["Name"].lower()}.gif', width = 150)
            st.write(f'Pokédex #: {pokemon["#"]}')
            st.write(f'Predicted Pokémon: {pokemon["Name"]}')
            st.write(f'Pokédex Entry: {pokemon["Entry"]}')
            st.audio(f'../assets/cries/{pokemon["#"]}.ogg', format = "audio/ogg")

        # PLOT POKEMON STATS
        data = {
            'Category': ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp Def', 'Speed'],
            'Values' : [pokemon["HP"], pokemon["Attack"], pokemon["Defense"],
                        pokemon["Sp. Atk"], pokemon["Sp. Def"], pokemon["Speed"]]
        }

        stats_df = pd.DataFrame(data)
        with col2:
            chart = alt.Chart(stats_df).mark_bar().encode(y = alt.Y('Category:N', sort = None, axis = alt.Axis(labelFontSize = 16)),
                                                        x = alt.X("Values:Q", scale = alt.Scale(domain = [0, 125]), axis = None),
                                                        color = alt.Color('Values:Q', scale = alt.Scale(domain = [20, 125], range = ['red', 'green'], type = 'linear'),
                                                                            legend = None)
            ).properties(width = 800, height = 350)
            st.title("Stats")
            st.altair_chart(chart, use_container_width = True)

