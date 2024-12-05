#!/bin/bash
docker container stop gr300_local
docker container rm gr300_local
docker run -it --gpus all --name gr300_local \
-v /mnt/d/Works/jupyterlab:/mnt/workspace -p 28888:18888 \
--ipc=host -m 200g --memory-swap 300g --shm-size="32g" gr300/local \
        bash -c "
			Xvfb :99 -screen 0 1024x768x16 & \
			jupyter notebook --ip 0.0.0.0 --port 18888 \
			--allow-root --no-browser --notebook-dir /mnt/workspace
        "

bash
sh

