# 👾	 Pokémon Classifier (Gen 1)

A deep learning-powered web app that identifies **1st generation Pokémon** from an image!

---
## 🌐 Live Demo

🚀 Check out the deployed app here:  
[**🔗 Pokémon Classifier (Streamlit Cloud)**](https://kantodex1017.netlify.app/)

> Upload an image of a Gen 1 Pokémon and the model will predict which Pokémon it is. 

---

## 📦 Overview

This project uses **transfer learning** with a ResNet-18 model to classify Pokémon from uploaded images. After training and fine-tuning, the model is integrated into a **React + FastAPI** web app that predicts the Pokémon species from an uploaded image.

---

## 📊 Dataset

### 📈 Stats
- **Source**: Scraped from [Pokémon Database](https://pokemondb.net)
- Contains HP, Attack, Defense, Speed, Special, and Total stats for all Gen 1 Pokémon.

### 🖼️ Images
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)
- Images for all 151 Gen 1 Pokémon in various styles.


## 🧠 Model

- **Architecture**: `ResNet-18` from `torchvision.models`
- **Modifications**:
  - Replaced the final fully-connected layer to output `150` classes (for 150 Pokémon images)
- **Training**:
  - Optimizer: `Adam`
  - Loss Function: `CrossEntropyLoss`
  - Accuracy Achieved: **~89%** on test set

---

## ⚠️ Limitations

While the KantoDex Classifier performs well within its design scope, there are a few known limitations:

- ❌ **Excludes Nidoran♂ and Nidoran♀**: Due to character encoding issues and image/sprite naming inconsistencies, these Pokémon were excluded from the dataset.
- 🧬 **Only Supports 1st Generation Pokémon**: The model is trained exclusively on the original 150 Pokémon, meaning it will not recognize Pokémon from later generations..
- ❌ **Excludes Stats and Information of Pokémon**: Work in Progress.

--- 
## 🚀 Getting Started

### 1. Install dependences:
```
pip install -r requirements.txt
```

### 2. Prepare the model:
```
cd backend
jupyter notebook notebooks/main.ipynb
```

### 3. Start the backend:
```
uvicorn main:app --reload
```

### 4. In a new terminal, create environment file with local backend and start frontend:
Make sure to include ```VITE_API_URL=http://127.0.0.1:8000``` in ```.env.dev```
```
cd frontend
cp .env.dev
npm run dev:dev
```
