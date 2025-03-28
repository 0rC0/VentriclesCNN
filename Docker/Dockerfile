FROM python:3.11-slim

# Install Miniconda
RUN apt-get update && apt-get install -y wget bzip2 && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/miniconda && \
    rm /miniconda.sh
ENV PATH="/opt/miniconda/bin:${PATH}"

# Copy the environment.yml file
COPY nnunet_conda_env.yaml /tmp/environment.yml
# Create the conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy
ENV PATH /opt/miniconda/envs/nnunet/bin:$PATH

# Install OSF.io client to download the models
RUN pip install osfclient


#### .... ToDo.....

# Expose a volume for input/output
VOLUME ["/input", "/output"]

# Set the working directory
WORKDIR /input

