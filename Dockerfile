FROM python:3.11

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD solara run app.py --host 0.0.0.0 --port 7860