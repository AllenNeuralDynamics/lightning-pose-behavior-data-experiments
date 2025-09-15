# hash:sha256:3ed0de2fbe9de993f0bbafc4a5e3a6030421ac2dc7653e0d1f799bee573b3292
FROM registry.codeocean.allenneuraldynamics.org/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libsm6=2:1.2.3-1 \
        libxext6=2:1.3.4-0ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=conda-forge \
        jupyterlab==3.5.3 \
        ipywidgets \
        protobuf==3.20.* \
    && conda clean -ya

ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

COPY postInstall /
RUN /postInstall
