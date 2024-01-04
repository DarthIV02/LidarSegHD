#FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Just in case we need it
ENV DEBIAN_FRONTEND='noninteractive'

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC

RUN apt update && apt-get upgrade -y
RUN apt install -y python3 python3-dev python3-pip
RUN pip3 install --upgrade pip
RUN apt install -y git wget unzip 
RUN apt-get install -y vim tmux
RUN apt install -y build-essential

# get newer version of cmake
RUN wget https://apt.kitware.com/kitware-archive.sh
RUN chmod +x kitware-archive.sh
RUN ./kitware-archive.sh

## install conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh && \
    bash Anaconda3-2023.03-1-Linux-x86_64.sh -b

ENV PATH=$PATH:/root/anaconda3/bin/

## Install Cylinder 3D
WORKDIR /
RUN git clone --recursive --depth 1 https://github.com/xinge008/Cylinder3D /root/Cylinder3D
WORKDIR /root/Cylinder3D
COPY . .

RUN conda create -n cylinder -y && \
    conda init bash

# make RUN commands use the new environment
SHELL ["conda", "run", "--no-capture-output", "-n", "cylinder", "/bin/bash", "-c"]
RUN echo 'conda activate cylinder' >> /root/.bashrc

RUN yes | pip install --upgrade pip

RUN yes | pip install numpy
RUN yes | pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/
RUN yes | pip install pyyaml
RUN yes | pip install cython
RUN yes | conda install opencv
#RUN yes | pip install opencv-python
#RUN yes | pip install nuscenes-devkit==1.1.6
RUN yes | pip install numba
RUN yes | pip install strictyaml
RUN yes | pip install tqdm
RUN yes | pip install spconv-cu117

RUN apt-get install -y ffmpeg libsm6 libxext6

# Torch scatter
RUN yes | pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html

# SpConv
# need to use SpConv 1 and hotfixes
# see: https://github.com/traveller59/spconv/issues/58
#RUN git clone --recursive https://github.com/traveller59/spconv.git /spconv
#RUN wget -O hotfixes.zip https://github.com/traveller59/spconv/files/4658204/hotfixes.zip
#RUN unzip hotfixes -d /hotfixes
#WORKDIR /spconv
#RUN git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634 
# needs to be done before we can apply the patches
#RUN git config --global user.email "test@test.com"
#RUN git config --global user.name "Test"
#RUN git am /hotfixes/0001-fix-problem-with-torch-1.4.patch 
#RUN git am /hotfixes/0001-Allow-to-specifiy-CUDA_ROOT-directory-and-pick-corre.patch
#RUN yes | apt install -y libboost-all-dev cmake
ENV CUDA_ROOT=/usr/local/cuda
#RUN yes | pip install wheel
#WORKDIR /spconv/third_party
#RUN git clone https://github.com/pybind/pybind11.git
#WORKDIR /spconv

# MAKE SURE TO RUN THIS COMMANDS AFTER in spconv
# vim include/spconv/nms.h
# include <iostream>
#RUN python setup.py bdist_wheel
#WORKDIR /spconv/dist
#RUN pip install *.whl