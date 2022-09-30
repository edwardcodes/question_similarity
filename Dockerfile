
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# ARG DEBIAN_FRONTEND=noninteractive

#set up environment
RUN apt-get update
RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip

# Installing Pytorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
# "run", "--host=0.0.0.0"
CMD ["python3", "main.py"]

