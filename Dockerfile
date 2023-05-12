# We will use python:3.10-alpine as the base image for building the Flask container
FROM amd64/python:3.10-bullseye
# ENVVAR for the model
ENV FINALMODEL=/app/generator.h5
# Specifies the working directory where the Docker container will run
WORKDIR /app
# Copying all the application files to the working directory
COPY web_app/ .
COPY requirements.txt .
COPY models/generatorV1.h5 generator.h5
# Install all the dependencies required to run the Flask application
RUN pip install --no-cache-dir -r requirements.txt
# Expose the Docker container for the application to run on port 8443 with ssl/HTTPS
EXPOSE 8443
# The command required to run the Dockerized application
CMD ["python", "/app/app.py"]