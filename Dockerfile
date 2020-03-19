FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install -U Pillow scipy pytest
RUN pip install -r requirements.txt

WORKDIR /dct
