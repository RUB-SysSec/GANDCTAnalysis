FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install -U Pillow scipy pytest
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt install git -y
RUN pip install git+https://github.com/CNOCycle/cleverhans.git@feature/tf2.x 

WORKDIR /dct
