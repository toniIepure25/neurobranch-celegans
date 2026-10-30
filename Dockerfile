# Research container for NeuroBranch.
#
# Build:
#   docker build -t neurobranch:1.3.2 .
# Run the app with reports mounted read-only:
#   docker run --rm -p 8501:8501 -v "$PWD/reports:/app/reports:ro" neurobranch:1.3.2
#
# The image does not contain the Kato or Randi data. Mount data/raw and reports/
# when reproducing analyses; the licence audit applies to both.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 neurobranch
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .

COPY app ./app
COPY schemas ./schemas
COPY scripts ./scripts
COPY reports ./reports

USER neurobranch
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
