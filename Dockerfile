FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean
# RUN apt install default-jre
RUN pip install transformers==4.12.5
RUN pip install vncorenlp
RUN pip install pandas
RUN pip install ujson
RUN pip install mlflow
RUN pip install tensorboard
