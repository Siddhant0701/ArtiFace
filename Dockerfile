# We will use python:3.10-alpine as the base image for building the Flask container
FROM amd64/python:latest

# ENVVAR for the model
ENV FINALMODEL=generator_epoch_190.h5

# It specifies the working directory where the Docker container will run
WORKDIR /app

# Copying all the application files to the working directory
COPY web_app/ ./web_app/
COPY models/generator_epoch_190.h5 .
COPY requirements.txt .

# Install all the dependencies required to run the Flask application
RUN pip install -r requirements.txt

# Expose the Docker container for the application to run on port 5000
EXPOSE 5000

# The command required to run the Dockerized application
CMD ["python", "/app/web_app/app.py"]
