#!/bin/bash
# We are overriding the entrypoint here, so we can use the docker image for testing and in the c2d environment at the same time
docker run -v "$(pwd)"/data:/data --entrypoint python load-diversity-c2d:main code/load-diversity-c2d.py
