FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt /srv/orchestrator/requirements.txt
RUN pip install --no-cache-dir -r /srv/orchestrator/requirements.txt

COPY . /srv/orchestrator/

EXPOSE 8002
CMD ["uvicorn", "orchestrator.app:app", "--host", "0.0.0.0", "--port", "8002"]
