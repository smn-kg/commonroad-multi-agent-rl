#! /bin/bash

# This shell script is used to automatically build the containers for the python image
# ------------------------------------------------------------------------------------

# Information
echo "Starting to build and push the containers for cps-rl. This may take some time."
echo "You have to be in the docker group to run this script successfully."
echo -e "\n  -- lrz gitlab login --"

# Docker gitlab login (You need to use access token for password)
docker login gitlab.lrz.de:5005

# Create cpp standalone and push to container registry
docker build --no-cache -f Dockerfile -t gitlab.lrz.de:5005/cps/cps-rl/safe-rl-autodrive/python_image:3.11 .
docker push gitlab.lrz.de:5005/cps/cps-rl/safe-rl-autodrive/python_image:3.11

# docker gitlab logout
docker logout gitlab.lrz.de:5005
