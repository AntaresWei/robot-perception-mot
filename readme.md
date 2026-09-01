

### 环境配置

Ubuntu: 22.04
CUDA: 11.8
GPU: NVIDIA GeForce RTX 4090

```
conda create -n robot-perception-mot python=3.10 -y

conda activate robot-perception-mot

pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

pip install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    opencv-python \
    open3d \
    pyyaml \
    tqdm \
    matplotlib \
    scikit-learn


pip install \
    filterpy \
    lap \
    motmetrics
```


