FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 AS app

WORKDIR /workspace

# define input arguments
ARG MINICONDA="Miniconda3-py38_4.9.2-Linux-x86_64.sh"
ARG USER_ID="1000"
ARG GROUP_ID="1000"

# set env variables
ENV LAYERJOT_HOME="/layerjot"
ENV PYTHONPATH="${LAYERJOT_HOME}/FGVC-PIM"
ENV DEBIAN_FRONTEND="noninteractive"

# install system packages
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
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
    unzip && \
    apt-get clean

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/${MINICONDA} \
    && echo "1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7 ${MINICONDA}" > sha1sums.txt \
    && sha256sum -c sha1sums.txt \
    && bash ${MINICONDA} -b -p miniconda
ENV PATH="/workspace/miniconda/bin:$PATH"

# install conda packages
RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# upgrade pip
RUN python -m pip install --upgrade pip

# Simple root password in case we want to customize the container
RUN echo "root:root" | chpasswd

# set up non-root user
RUN addgroup --gid "$GROUP_ID" user
RUN useradd -G video,audio -ms /bin/bash --uid "$USER_ID" --gid "$GROUP_ID" user
RUN echo "user:user" | chpasswd && adduser user sudo

# copy the code
COPY . .

# install requirements
RUN pip install -r requirements.txt && pip install -e .

# set the runtime user
USER user

# set the entrypoint
ENTRYPOINT [ "/bin/bash" ]
