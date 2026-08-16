FROM python:3.12-slim

WORKDIR /srv

# Build context is the REPO ROOT (deploy.sh: docker build -f orchestrator/Dockerfile .)
# so the image can ship both the orchestrator adapter and the channel-neutral
# prism_core package.
COPY orchestrator/requirements.txt /srv/orchestrator/requirements.txt
RUN pip install --no-cache-dir -r /srv/orchestrator/requirements.txt

COPY orchestrator/ /srv/orchestrator/
COPY prism_core/ /srv/prism_core/

EXPOSE 8002
CMD ["uvicorn", "orchestrator.app:app", "--host", "0.0.0.0", "--port", "8002"]
