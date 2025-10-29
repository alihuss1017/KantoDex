FROM python:3.11-slim


WORKDIR /app


COPY backend/ /app/
COPY backend/requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch==2.0.0+cpu torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


EXPOSE 8080


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
