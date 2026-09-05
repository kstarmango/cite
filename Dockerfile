# HF Spaces(Docker SDK)용. 로컬 개발은 이 파일 없이도 그대로 동작한다.
# 3.12 고정 이유: requirements의 numpy==2.5.1 이 cp311 휠을 제공하지 않는다(최대 2.4.6).
FROM python:3.12-slim

# libgomp1 = onnxruntime(OpenMP) 런타임 의존. 없으면 fastembed import 단계에서 죽는다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces 컨테이너는 uid 1000으로 돈다. 홈을 잡아줘야 모델 캐시를 쓸 수 있다.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

USER user

# 모델 2개를 빌드 시점에 받아 이미지에 굽는다.
# 런타임에 받게 두면 첫 요청이 수십 초 걸리고, Space 재시작마다 반복된다.
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

COPY --chown=user . .

# HF Spaces 고정 포트
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
