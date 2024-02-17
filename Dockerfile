FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 AS app

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA="Miniconda3-py310_23.1.0-1-Linux-x86_64.sh"

ARG USER_ID=1000
ARG GROUP_ID=1000

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    lsb-release \
    protobuf-compiler \
    vim \
    wget \
    gcc \
    libglib2.0-0 \
    unzip

# Simple root password in case we want to customize the container
RUN echo "root:root" | chpasswd
RUN echo "Group ID $GROUP_ID"
RUN echo "User id $USER_ID"
RUN addgroup --gid $GROUP_ID user

RUN useradd -G video,audio -ms /bin/bash --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update && \
    apt-get -y install sudo

RUN echo "user:user" | chpasswd && adduser user sudo

WORKDIR /workspace

COPY . /workspace

ENV LAYERJOT_HOME="/layerjot"
ENV PATH=$PATH
ENV PYTHONPATH=$LAYERJOT_HOME/FGVC-PIM

RUN wget https://repo.anaconda.com/miniconda/${MINICONDA} \
    && bash ${MINICONDA} -b -p miniconda

ENV PATH=/workspace/miniconda/bin:$PATH

RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

RUN cd /workspace && pip install -r requirements.txt \
    && pip install -e . \
    && python -m pip install --upgrade pip

USER user
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

