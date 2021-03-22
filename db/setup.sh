#!/bin/bash

service postgresql restart
user=`whoami`
sudo -u postgres bash -c "createuser -s $user"
createdb dr
psql -d dr -f db/create.sql

# Setup password (for Docker)
psql -c "ALTER ROLE $user WITH PASSWORD 'pass';" dr

# NOTE: Install different version of numpy for hdbscan
pip3 install numpy==1.20.1
