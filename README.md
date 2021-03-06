# FPGA-GPU-Cluster
The FPGA/GPU cluster is a cloud-based, remotely accessible compute infrastructure specifically designed to accelerate compute intensive applications, such as machine learning training and inference, video processing, financial computing, database analytics networking and bioinformatics. Latest state of the art acceleration technologies including the Alveo FPGAs, and Tesla V100 GPUs, closely coupled with server processors constitute the backbone of this cluster. The software stack consists of a complete ecosystem of machine learning frameworks, libraries and runtime targeting heterogeneous computing accelerators.
![image](https://user-images.githubusercontent.com/9284879/119378393-3c6aaf00-bc8c-11eb-92df-92df33ddc13d.png)

# FPGA/GPU Cluster Software Stack
The FPGA/GPU cluster supports three the most commonly used deep learning frameworks, namely, TensorFlow, Caffe and Pytorch. These frameworks provide a high-level abstraction layer for deep learning architecture specification, model training, tuning, testing, and validation. The software stack also includes various machine learning vendor-specific libraries that provide dedicated computing functions tuned for specific hardware architecture, delivering the best possible performance/power figure.

![image](https://user-images.githubusercontent.com/9284879/119378658-7b990000-bc8c-11eb-8fb0-9ab17804d898.png)

# Quick Start Guide: Remote Access to the FPGA/GPU Cluster
This quick guide provides instructions on how to reserve, access, manage, and use the FPGA/GPU cluster nodes through the CMC cloud environment interface.
https://www.cmc.ca/qsg-fpga-gpu-cluster/

## Copy the necessary files to your home directory
> cp -R /CMC/accelerator/GetStarted/ .

> cd GetStarted

# Test Tensorflow
## Test Tensorflow on terminal
> source /CMC/accelerator/anaconda3/bin/activate

> conda activate /CMC/accelerator/anaconda3/envs/tensorflow
 
> python test_tensorflow.py
 
>	conda deactivate
## Test Tensorflow on Jupyter
### Register your Environment by using the following command:
> python -m ipykernel install --user --name tensorflow --display-name "Python 3.7 (tensorflow)"
### Test your Environment by launching Jupyter notebook:
> jupyter notebook

# Test Pytorch
## Test Pytorch on terminal
> source /CMC/accelerator/anaconda3/bin/activate

> conda activate /CMC/accelerator/anaconda3/envs/torch

> python test_pytorch.py
## Test Pytorch on Jupyter
### Register your Environment by using the following command:
> python -m ipykernel install --user --name pytorch --display-name "Python 3.7 (pytorch)"
### Test your Environment by launching Jupyter notebook:
> jupyter notebook                            
