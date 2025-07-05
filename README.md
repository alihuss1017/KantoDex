# 🧠 Pokémon Classifier (Gen 1)

A deep learning-powered web app that identifies **1st generation Pokémon** from an image and displays relevant data like stats, sprite, and cry!

---

## 📦 Overview

This project uses **transfer learning** with a ResNet-18 model to classify Pokémon from uploaded images. After training and fine-tuning, the model is integrated into an interactive **Streamlit** app that:

- Predicts the Pokémon species from an uploaded image
- Displays its animated sprite
- Plays its cry sound
- Shows its base stats

---

## 📊 Dataset

### 📈 Stats
- **Source**: Scraped from [Pokémon Database](https://pokemondb.net)
- Contains HP, Attack, Defense, Speed, Special, and Total stats for all Gen 1 Pokémon.

### 🖼️ Images
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)
- Images for all 151 Gen 1 Pokémon in various styles.

### 🔊 Cries
- **Source**: [The Sounds Resource](https://www.sounds-resource.com/3ds/pokemonultrasunultramoon/sound/9547/)
- Cry sound files for each Pokémon stored in `assets/cries/`.

---

## 🧠 Model

- **Architecture**: `ResNet-18` from `torchvision.models`
- **Modifications**:
  - Replaced the final fully-connected layer to output `150` classes (for 150 Pokémon images)
- **Training**:
  - Optimizer: `Adam`
  - Loss Function: `CrossEntropyLoss`
  - Accuracy Achieved: **~89%** on test set

---

## 🖥️ App Functionality

The app is built with **Streamlit** and allows users to:

1. **Upload an image** of a Pokémon
2. The model **predicts the species**
3. The app:
   - Displays an **animated sprite** of the predicted Pokémon
   - Plays the **cry** sound of that Pokémon
   - Shows its **base stats** from the Pokédex

## 🚀 How to Run

This project can be run **locally** with a Python virtual environment.

### 1. 🔁 Clone the Repository
```bash
git clone https://github.com/alihuss1017/KantoDex.git
cd KantoDex
```

### 2. 🐍 Create and Activate a Virtual Environment (Python 3.10+)
```
python3 -m venv myenv
source myenv/bin/activate       # macOS/Linux
# OR
myenv\Scripts\activate          # Windows
```

### 3. 📦 Install Required Dependencies
```
pip install -r requirements.txt
```

### 4. 🚀 Launch the Streamlit App
```
cd src
streamlit run app.py
```
---

## ⚠️ Limitations

While the KantoDex Classifier performs well within its design scope, there are a few known limitations:

- ❌ **Excludes Nidoran♂ and Nidoran♀**: Due to character encoding issues and image/sprite naming inconsistencies, these Pokémon were excluded from the dataset.
- 🧬 **Only Supports 1st Generation Pokémon**: The model is trained exclusively on the original 150 Pokémon, meaning it will not recognize Pokémon from later generations..



