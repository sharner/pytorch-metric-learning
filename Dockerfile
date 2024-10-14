FROM nvidia/cuda:12.3.1-devel-ubuntu20.04 AS app

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ARG DEBIAN_FRONTEND=noninteractive

# install miniconda
ENV CONDA_HOME=/opt/conda
ARG conda_ver="py311_24.5.0-0-Linux-x86_64"

#ARG user_id=${USER_ID}
#ARG group_id=${GROUP_ID}
# hack.  Is there a better way?  1002=forest 1004=data on Thor
ARG user_id=1002
ARG group_id=1004

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

# Install dependencies for VS Code so that we can do VS Code Tunnels
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    libc-ares-dev \
    liblz4-dev \
    libfreetype6-dev \
    ninja-build \
    gnome-desktop-testing \
    libre2-dev \
    libasound2-dev \
    libpulse-dev \
    libaudio-dev \
    libjack-dev \
    libsndio-dev \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxfixes-dev \
    libxi-dev \
    libxss-dev \
    libxkbcommon-dev \
    libnspr4 \
    libnss3 \
    libgbm-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libdbus-1-dev \
    libibus-1.0-dev \
    libudev-dev \
    fcitx-libs-dev \
    xdg-utils \
    x11-apps \
    sudo

# Simple root password in case we want to customize the container
RUN echo "root:root" | chpasswd
RUN echo "Group ID $group_id"
RUN echo "User id $user_id"
ENV APP_USER="user"
RUN addgroup --gid ${group_id} ${APP_USER}

RUN useradd -G video,audio -ms /bin/bash --uid $user_id --gid $group_id user

RUN echo "user:user" | chpasswd && adduser user sudo

WORKDIR /workspace
COPY . /workspace

ENV LAYERJOT_HOME="/layerjot"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-${conda_ver}.sh \
    && bash Miniconda3-${conda_ver}.sh -b -p ${CONDA_HOME}
ENV PATH=/home/user/.local/bin:${CONDA_HOME}/bin:$PATH

# RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch faiss-gpu=1.8.0
RUN conda install -y cudatoolkit=11.4 faiss-gpu=1.8.0 -c nvidia -c pytorch

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# This fails in build step, so you can run it by hand in the container.
# Install code CLI
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz \
    && tar -xf vscode_cli.tar.gz \
    && rm -f vscode_cli.tar.gz \
    && mv code /usr/local/bin/code

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

RUN cd /workspace && pip install -r requirements.txt \
    && pip install -e . \
    && python -m pip install --upgrade pip

USER user
ENV PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin

ENTRYPOINT [ "bash" ]
