# ============================
# 1️⃣ Use an official Python base image
# ============================
FROM python:3.11-slim

# ============================
# 2️⃣ Set working directory inside the container
# ============================
WORKDIR /app

# ============================
# 3️⃣ Copy dependency list and install them
# ============================
COPY requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 4️⃣ Copy the rest of your FastAPI app files
# ============================
COPY main.py ./main.py
COPY models/pokemodel.pt ./models/pokemodel.pt
COPY data/pokedex_info.csv ./data/pokedex_info.csv

# ============================
# 5️⃣ Expose the port FastAPI runs on (default 8000)
# ============================
EXPOSE 8000

# ============================
# 6️⃣ Start FastAPI server with uvicorn
# ============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
