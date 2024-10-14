FROM ghcr.io/railwayapp/nixpacks:ubuntu-1727136237

WORKDIR /app/

COPY . /app/

RUN python -m venv --copies /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "app:app"]