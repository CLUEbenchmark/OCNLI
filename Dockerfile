FROM tensorflow/tensorflow:1.12.0-gpu-py3
LABEL repository="OCNLI"
WORKDIR /stage/allennlp

# Install requirements
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Copy code
COPY ocnli/ ocnli/

ENTRYPOINT []
CMD ["/bin/bash"]