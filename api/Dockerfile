FROM python:3.12-slim

# System deps for music21 and FluidSynth (audio rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fluidsynth \
    libfluidsynth3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY decimus/ decimus/
COPY fairseq/ fairseq/
COPY musicBPE/ musicBPE/
COPY preprocess/ preprocess/
COPY encoding.py .
COPY run_web.py .
COPY main.py .

# Install package and all dependencies
RUN pip install --no-cache-dir .

# Railway injects PORT; default to 8000
ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn decimus.web.app:app --host 0.0.0.0 --port ${PORT}"]
