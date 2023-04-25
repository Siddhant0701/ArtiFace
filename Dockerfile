# We will use python:3.10 as the base image for building the Flask container
#FROM python:3.10.8-bullseye	

# ENVVAR for the model
#ENV FINALMODEL=/app/generator_epoch_190.h5

# It specifies the working directory where the Docker container will run
# WORKDIR /usr/src/app

# Copying all the application files to the working directory
#COPY web_app/ .
#COPY checkpoints/generator_epoch_190.h5 .
#COPY requirements.txt .

# Install all the dependencies required to run the Flask application
#RUN pip install --no-cache-dir -r requirements.txt

# Expose the Docker container for the application to run on port 5000
# EXPOSE 5000

#Change the working directory to the web_app folder
#WORKDIR /app/web_app

# The command required to run the Dockerized application
# CMD ["python", "/app/web_app/app.py"]





# We will use python:3.10-alpine as the base image for building the Flask container
FROM amd64/python:3.10-bullseye
# ENVVAR for the model
ENV FINALMODEL=/app/generator_epoch_190.h5
# It specifies the working directory where the Docker container will run
WORKDIR /app
# Copying all the application files to the working directory
COPY web_app/ .
COPY requirements.txt .
COPY checkpoints/generator_epoch_190.h5 .
# Install all the dependencies required to run the Flask application
RUN pip install --no-cache-dir -r requirements.txt
# Expose the Docker container for the application to run on port 5000
EXPOSE 5000
# The command required to run the Dockerized application
CMD ["python", "/app/app.py"]