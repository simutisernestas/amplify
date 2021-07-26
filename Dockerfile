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

ARG ROS_PKG=ros_base
ENV ROS_DISTRO=galactic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"] 

WORKDIR /tmp

# change the locale from POSIX to UTF-8
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US en_US.UTF-8 
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8


# 
# add the ROS deb repo to the apt sources list
#
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  wget \
  gnupg2 \
  lsb-release \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


# 
# install development packages
#
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  libbullet-dev \
  libpython3-dev \
  python3-colcon-common-extensions \
  python3-flake8 \
  python3-pip \
  python3-numpy \
  python3-pytest-cov \
  python3-rosdep \
  python3-setuptools \
  python3-vcstool \
  python3-rosinstall-generator \
  libasio-dev \
  libtinyxml2-dev \
  libcunit1-dev \
  libgazebo9-dev \
  gazebo9 \
  gazebo9-common \
  gazebo9-plugin-base \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# install some pip packages needed for testing
RUN python3 -m pip install -U \
  argcomplete \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest-repeat \
  pytest-rerunfailures \
  pytest

# 
# generate ROS source code workspace
# https://answers.ros.org/question/325245/minimal-ros2-installation/?answer=325249#post-id-325249
#
RUN mkdir -p ${ROS_ROOT}/src && \
  cd ${ROS_ROOT} && \
  rosinstall_generator --deps --rosdistro ${ROS_DISTRO} ${ROS_PKG} \
  launch_xml \
  launch_yaml \
  launch_testing \
  launch_testing_ament_cmake \
  demo_nodes_cpp \
  demo_nodes_py \
  example_interfaces \
  camera_calibration_parsers \
  camera_info_manager \
  cv_bridge \
  v4l2_camera \
  vision_opencv \
  vision_msgs \
  image_transport \
  image_geometry \
  image_pipeline \
  > ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
  cat ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
  vcs import src < ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall

# download unreleased packages
#RUN git clone --branch ros2 https://github.com/ros-perception/vision_msgs ${ROS_ROOT}/src/vision_msgs && \
#    git clone --branch ${ROS_DISTRO} https://github.com/ros2/demos demos && \
#    cp -r demos/demo_nodes_cpp ${ROS_ROOT}/src && \
#    cp -r demos/demo_nodes_py ${ROS_ROOT}/src && \
#    rm -r -f demos


# 
# install dependencies using rosdep
#
RUN apt-get update && \
  cd ${ROS_ROOT} && \
  rosdep init && \
  rosdep update && \
  rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y --skip-keys "libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv" && \
  rm -rf /var/lib/apt/lists/* && \
  apt-get clean

# 
# build it!
#
RUN cd ${ROS_ROOT} && colcon build --symlink-install

#
# setup entrypoint
#
COPY ./packages/ros_entrypoint.sh /ros_entrypoint.sh

RUN sed -i \
  's/ros_env_setup="\/opt\/ros\/$ROS_DISTRO\/setup.bash"/ros_env_setup="${ROS_ROOT}\/install\/setup.bash"/g' \
  /ros_entrypoint.sh && \
  cat /ros_entrypoint.sh

RUN echo 'source ${ROS_ROOT}/install/setup.bash' >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
WORKDIR /
