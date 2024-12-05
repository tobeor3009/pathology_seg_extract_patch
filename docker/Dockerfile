# [필요한 docker hub image 주소 or 버전]
# https://hub.docker.com/r/pytorch/pytorch 여기를 찾아봐야함
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# 필수 : 시스템 설정 - 타임존
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/ /etc/localtime && echo  > /etc/timezone
RUN apt-get update -y
RUN apt-get install vim net-tools -y
# 필수 : 추가 패키지
# Solve python module opency-python dependency
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libxrender1 -y
RUN apt-get update && apt-get install libglib2.0-0 -y --fix-missing
# 사용자 선택 : 필요한 라이브러리 설치
# 필요한 패키지 apt-get install -y 으로 나열
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install jupyter jupyterlab
RUN pip3 install notebook
# opencv 설치를 겸하고 있어 가장 느린 코드
RUN pip3 install albumentations==1.3.0
# 호환되는 버전 찾기위해 여기 https://www.tensorflow.org/install/source?hl=ko#gpu
# https://pypi.org/project/tensorflow/#history
RUN pip3 install tensorflow==2.14.1 tensorflow_addons
RUN pip3 install tensorboard monai wandb einops
RUN pip3 install numpy matplotlib tqdm pandas scikit-learn scikit-image seaborn
RUN pip3 install SimpleITK nibabel pydicom # for medical imaging
RUN pip3 install openpyxl progressbar natsort
RUN pip3 install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 # stylegan-ada-pytorch requirements
RUN pip3 install shapely mahotas lxml imutils
RUN pip3 install imagecodecs torchsummary
RUN pip3 install segmentation_models segmentation_models_pytorch 
RUN pip3 install xlrd==1.2.0
RUN pip3 install setuptools==69.5.1
RUN pip3 install openai kaggle deepspeed==0.12.6
RUN pip3 install bitsandbytes plotly 
RUN pip3 install monai monai-generative
RUN pip3 install volumentations-3D vtk
RUN apt-get install git cmake build-essential -y
RUN apt-get install curl vim ca-certificates libjpeg-dev libpng-dev -y
RUN apt-get install libopenexr-dev libglew-dev libopencv-dev -y
RUN apt-get install libgoogle-glog-dev libgflags-dev libatlas-base-dev -y
# deepspeed's mpi4py dependency
RUN conda update conda -y
RUN conda init
RUN conda install mpich mpi4py ninja -y
RUN conda install -c conda-forge openslide-python -y
RUN pip3 install pip==24.0
ENV MAX_JOBS=4
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
ENV MAX_JOBS=N
RUN pip3 install https://github.com/VincentStimper/mclahe/archive/numpy.zip
RUN apt-get install xvfb -y
ENV DISPLAY=:99
#############################################
## Recommand Add your Custom pip install here
#############################################
RUN pip3 install jupyterlab-classic
RUN pip3 install jupyterlab-theme-solarized-dark
RUN pip3 install jupyterthemes
RUN jt -t monokai -fs 115 -nfs 125 -tfs 115 -dfs 115 -ofs 115 -cursc r -cellw 80% -lineh 115 -altmd  -kl -T -N
# 불필요한 pip 라이브러리 삭제 (자동완성 기능 - 느림)
RUN pip3 uninstall jedi -y
RUN jupyter lab --generate-config
# set jupyter notebook's password as root
RUN echo "c.NotebookApp.password = u'sha256:50cd326c063b:6a4c1d4cfc5ec7ecd9f7dcce986fb08fca9f3ac808dd82d365d79cc445dd498e'">>/root/.jupyter/jupyter_notebook_config.py