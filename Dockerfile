FROM tensorflow/tensorflow:2.3.0-jupyter
WORKDIR /app
COPY . /app
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libhdf5-dev \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-dev
RUN pip3 --no-cache-dir install -r requirements.txt
EXPOSE 8080
CMD [ "python", "classify.py" ]
