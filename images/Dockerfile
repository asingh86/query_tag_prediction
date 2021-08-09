FROM python:3.8-slim-buster

COPY ../requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD jupyter notebook --port=8080 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.max_buffer_size=4000000000 --ResourceUseDisplay.mem_warning_threshold=0.2