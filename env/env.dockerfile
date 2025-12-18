FROM docker.io/library/ubuntu:24.04
WORKDIR /
ENV TZ=America/Vancouver
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl wget make binutils nano unzip ca-certificates && \
        update-ca-certificates && rm -rf /var/lib/apt/lists/* 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /Miniconda3-latest-Linux-x86_64.sh && \
    /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm -rf /Miniconda3-latest-Linux-x86_64.sh && \
    /miniconda3/bin/conda init bash && /miniconda3/bin/conda config --set auto_activate_base false && \
    chmod -R 777 /miniconda3
WORKDIR /
ENV ENV_FOLDER=/env
SHELL ["/bin/bash", "-c"] 
COPY ./env/install_env.sh /root/install_env.sh
RUN export PATH="/miniconda3/bin:$PATH" && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    bash /root/install_env.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --mipgaussian --kaolin --nvdiffrast --demo && \
    rm -rf /root/.cache/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xz-utils libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libjpeg-dev && rm -rf /var/lib/apt/lists/*
RUN wget 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz' -P /opt && \
        tar -xvf /opt/blender-3.0.1-linux-x64.tar.xz -C /opt && \
        rm -rf /opt/blender-3.0.1-linux-x64.tar.xz
ENV SHELL=/bin/bash \
    PATH=/opt/conda/envs/trellis/bin:/opt/conda/condabin/conda/bin:$PATH \
    CONDA_PREFIX=/miniconda3/envs/trellis 
