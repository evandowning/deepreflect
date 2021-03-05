FROM python:3.7

workdir /opt/app

# Install python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy repo contents to docker
COPY . /opt/app

# Set entrypoint
ENTRYPOINT ["python", "main.py"]
