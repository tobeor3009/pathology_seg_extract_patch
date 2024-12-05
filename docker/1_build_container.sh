#!/bin/bash
docker build -t gr300/local . -f Dockerfile --memory=32g --memory-swap=64g

