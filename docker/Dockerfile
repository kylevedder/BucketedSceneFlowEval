FROM kylevedder/cudagl:11.8.0-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
# Set the timezone info because otherwise tzinfo blocks install
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make g++ unzip apt-utils

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda


ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
RUN conda install -y numpy=1.26 python=3.11 pytorch==2.3 torchvision torchaudio pytorch-cuda=11.8 fvcore iopath -c pytorch -c nvidia -c fvcore -c iopath -c conda-forge
# Install nuscenes-devkit, forked to get rid of the matplotlib pegged dependency
RUN conda install -y matplotlib=3.8
RUN apt install unzip -y
RUN pip install iopath fvcore pandas opencv-python pyquaternion pyarrow pytest av2 black isort

RUN git clone https://github.com/kylevedder/nuscenes-devkit.git /nuscenes-devkit
WORKDIR /nuscenes-devkit
RUN pip install -e setup

# Latest Open3D, because 0.18.0 has a bunch of bugs related to missing visualization functions
RUN pip install https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-0.18.0+18a47ef-cp311-cp311-manylinux_2_31_x86_64.whl

ENV PYTHONPATH=/bucketed_scene_flow_eval:${PYTHONPATH}
COPY bashrc /root/.bashrc
WORKDIR /bucketed_scene_flow_eval