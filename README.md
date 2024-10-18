# Ubuntu_env
Ubuntu environment

Docker pull command, (change /path/to/folder to your acutal folder

```
sudo docker run --name ubuntu_env -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /path/to/folder:/home/marco/work_dir --env="DISPLAY=$DISPLAY" --privileged --shm-size 2g -it --user=marco --rm --workdir=/home/marco/work_dir marcosterlo/ubuntu_env:latest bash
```

My command with my folder:
```
sudo docker run --name ubuntu_env -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /home/marco/SSD/Uni\ offline/Working\ dir:/home/marco/work_dir --env="DISPLAY=$DISPLAY" --privileged --shm-size 8g -it --user=marco --rm --workdir=/home/marco/work_dir marcosterlo/ubuntu_env:latest bash
```

Docker commit command:
```
sudo docker commit ubuntu_env marcosterlo/ubuntu_env:latest
```

Docker push command:
```
sudo docker push marcosterlo/ubuntu_env:latest
```

# CONDA ALTERNATIVE
Install the following dependencies inside conda environment to have the same verison of the packages:
```
pip3 install numpy torch stable_baselines3 cvxpy control matplotlib scipy mosek
```
