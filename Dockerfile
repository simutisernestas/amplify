# modified https://github.com/microsoft/onnxruntime/blob/master/dockerfiles/Dockerfile.tensorrt
FROM nvcr.io/nvidia/tensorrt:20.12-py3

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=master

RUN apt-get update &&\
  apt-get install -y sudo git bash unattended-upgrades
RUN unattended-upgrade

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.20.3-linux-x86_64/bin:/opt/miniconda/bin:${PATH}

# Prepare onnxruntime repository & build onnxruntime with TensorRT
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
  /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
  cp onnxruntime/docs/Privacy.md /code/Privacy.md &&\
  cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
  cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
  cd onnxruntime &&\
  /bin/sh ./build.sh --parallel --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_shared_lib --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
  pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
  cd .. &&\
  rm -rf cmake-3.20.3-Linux-x86_64

RUN cd /code/onnxruntime/build/Linux/Release && make install

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 python3-opencv
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt update && apt install -y libopencv-dev

# FROM ros:galactic

# # install ros package
# RUN apt-get update && apt-get install -y \
#       ros-${ROS_DISTRO}-demo-nodes-cpp \
#       ros-${ROS_DISTRO}-demo-nodes-py \
#       python3-pip && \
#     rm -rf /var/lib/apt/lists/*

# ENV ROS_DOMAIN_ID=81

# WORKDIR /root/
# COPY --from=0 /code/onnxruntime/build/Linux/Release/dist/*.whl .
# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt 

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.20.3-linux-x86_64/bin:/opt/miniconda/bin:${PATH}

# RUN apt-get update &&\
#   apt-get install -y sudo git bash unattended-upgrades
# RUN unattended-upgrade

# RUN pip install --upgrade pip && pip install *.whl

# # launch ros package
# CMD ["ros2", "launch", "demo_nodes_cpp", "talker_listener.launch.py"]