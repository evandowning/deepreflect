FROM debian:buster

RUN apt update && apt upgrade -y && apt autoremove --purge -y
RUN apt update && apt install -y python3 python3-pip postgresql libpq-dev
RUN apt install -y p7zip-full gnome-shell dbus-x11 parallel
RUN apt install -y sudo
RUN pip3 install --upgrade pip

workdir /app

# Set Python alias
RUN rm /usr/bin/python
RUN ln -s python3 /usr/bin/python

# Install python requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy repo contents to docker
COPY . /app

# Install BinaryNinja
workdir /app/binja_setup
RUN bash ./setup.sh
workdir /app

# Set entrypoint
ENTRYPOINT ["python", "main.py"]
