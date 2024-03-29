FROM ogvalt/cuda:9.0-cudnn7.0-devel-ubuntu16.04

ENV DEBIAN_FRONTEND='noninteractive'

## fix out-of-date apt repo
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC

## apt installs
RUN apt update && apt-get upgrade -y
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y wget
RUN apt-get install -y git

## install conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh && \
    bash Anaconda3-2023.03-1-Linux-x86_64.sh -b

ENV PATH=$PATH:/root/anaconda3/bin/

## install repo / conda environment (repo instructions)
RUN git clone --depth=1 https://github.com/QingyongHu/SQN /root/SQN

# source ~/anaconda3/etc/profile.d/conda.sh <- If installing anaconda in terminal

RUN conda create -n sqn python=3.5 -y && \
    conda init bash

COPY . .

RUN dpkg -i libcudnn7_7.6.5.32-1+cuda9.0_amd64.deb
RUN dpkg -i libcudnn7-dev_7.6.5.32-1+cuda9.0_amd64.deb
RUN dpkg -i libcudnn7-doc_7.6.5.32-1+cuda9.0_amd64.deb

RUN sh /root/SQN/tf_ops/3d_interpolation/tf_interpolate_compile.sh

WORKDIR /root/SQN

# make RUN commands use the new environment
SHELL ["conda", "run", "--no-capture-output", "-n", "sqn", "/bin/bash", "-c"]

RUN yes | pip install --default-timeout=100 protobuf==3.19.4 && \
yes | pip install --default-timeout=100 tensorflow==1.11 && \
yes | pip install -r helper_requirements.txt && \
yes | pip install --default-timeout=100 matplotlib && \
yes | pip install --default-timeout=100 seaborn && \
yes | pip install  --default-timeout=100 tensorflow-gpu==1.11 && \ 
yes | apt-get install p7zip-full && \ 
yes | apt-get install vim && \
yes | apt-get install tmux && \
sh compile_op.sh

RUN echo 'conda activate sqn' >> /root/.bashrcpip