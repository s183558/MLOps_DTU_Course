# Base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# All the needed files and folders for the docker container (keep it as small as possible)
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY MLOps_DTU_project/ MLOps_DTU_project/
COPY data/ data/

# Install all the requirement.
# We use "--no-cache-dir" to not store the pip downloads, which would make it easier to redownload, but it takes up more space.
# We want the image to be as small as possible, why we dont want the cache
WORKDIR /
RUN pip install . --no-cache-dir #(1)

# The entrypoint for our docker image. "-u" is to redirect all print()'s to our terminal.
ENTRYPOINT ["python", "-u", "MLOps_DTU_project/train_model.py"]

# Create the docker image by typing this
# docker build -f dockerfiles/trainer.dockerfile . -t trainer:latest
