FROM python:3.11-slim

WORKDIR /app

ENV TZ=Asia/Seoul

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt . 

RUN pip install -r requirements.txt  # 라이브러리 목록이 안 바뀌면 캐시 사용

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
