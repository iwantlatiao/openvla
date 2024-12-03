# Install environment

```shell
conda create -n openvla python=3.11
conda activate openvla
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
```

[Install Cuda Linux](https://zhuanlan.zhihu.com/p/520536351)

[Install Cuda Windows](https://blog.csdn.net/qq_50677040/article/details/132131346)

[PyTorch Error checking compiler version for cl](https://stackoverflow.com/questions/73264234/pytorch-error-checking-compiler-version-for-cl-cpp-extension-py)

[Cuda Error C1189](https://liujiahua.com/blog/2024/05/29/cpp-CudaErrorC1189/)

[building wheel slow](https://stackoverflow.com/questions/73698418/building-wheel-for-opencv-python-keeps-running-for-a-very-long-time)

[Forced install flash-attn](https://blog.csdn.net/a486259/article/details/142695690)

However flash-attn was not installed.

[Windows 10 ROS-Melodic](https://blog.csdn.net/weixin_43563233/article/details/112238082)

[Ubuntu ROS 2](http://dev.ros2.fishros.com/doc/Installation/Ubuntu-Install-Binary.html)

[Install OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html)

symbolic link

```bash
# to a file
mklink Link Target

# to a directory
mklink /D Link Target
mklink /D "F:\code\openvla\data\tabletop_dark_wood" "F:\tabletop_dark_wood"
```

docker setup

```bash
# image config
ENV DEBIAN_FRONTEND=noninteractive

# run a image
docker run -v F:/:/mnt -d -t -p 3000:3000 --gpus all --name openvla_1126 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# exec bash
docker start openvla_1126
docker exec -it openvla_1126 /bin/bash

# install packages
apt update
apt install git
apt install wget

git config --global user.name "docker"
git config --global user.email "youremail"

pip install timm==0.9.10 tokenizers==0.19.1 transformers==4.40.1

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

wget http://fishros.com/install -O fishros && . fishros

# it seems that "agent_data.pkl" is not used in data processing.
# so we ignore the error "No module named 'sensor_msgs.msg._CameraInfo'"

# quantize llm in linux
cd ~
mkdir openvla
cd openvla
cp -r /mnt/code/openvla-win/vla-scripts ./vla-scripts
cp -r /mnt/code/openvla-win/saved_model/openvla-7b-model ./saved_model/openvla-7b-model
```

## Quantization Method

[HuggingFace docs](https://huggingface.co/docs/transformers/main/zh/main_classes/quantization)

```python
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
    load_in_4bit=True,
).to("cuda:0")
```