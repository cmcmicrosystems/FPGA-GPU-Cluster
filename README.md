## FPGA-GPU-Cluster
The FPGA/GPU cluster is a cloud-based, remotely accessible compute infrastructure specifically designed to accelerate compute intensive applications, such as machine learning training and inference, video processing, financial computing, database analytics networking and bioinformatics. Latest state of the art acceleration technologies including the Alveo FPGAs, and Tesla V100 GPUs, closely coupled with server processors constitute the backbone of this cluster. The software stack consists of a complete ecosystem of machine learning frameworks, libraries and runtime targeting heterogeneous computing accelerators.
![image](https://user-images.githubusercontent.com/9284879/119378393-3c6aaf00-bc8c-11eb-92df-92df33ddc13d.png)

## FPGA/GPU Cluster Software Stack
The FPGA/GPU cluster supports three the most commonly used deep learning frameworks, namely, TensorFlow, Caffe and Pytorch. These frameworks provide a high-level abstraction layer for deep learning architecture specification, model training, tuning, testing, and validation. The software stack also includes various machine learning vendor-specific libraries that provide dedicated computing functions tuned for specific hardware architecture, delivering the best possible performance/power figure.

![image](https://user-images.githubusercontent.com/9284879/119378658-7b990000-bc8c-11eb-8fb0-9ab17804d898.png)

## Quick Start Guide: Remote Access to the FPGA/GPU Cluster
This quick guide provides instructions on how to reserve, access, manage, and use the FPGA/GPU cluster nodes through the CMC cloud environment interface.
https://www.cmc.ca/qsg-fpga-gpu-cluster/

## Getting Started with TensorFlow and Docker: A Quick Guide

### Setting up the Environment

1. For the first time only, create a directory called `dockertmp`:
```bash
mkdir dockertmp
```
We will use this directory inside the running docker container. 

2. To use TensorFlow 1, run the following command:
```bash
yassine@uwaccel01:~$ docker run --gpus all -it -v $(pwd)/dockertmp:/mnt nvcr.io/nvidia/tensorflow:22.01-tf1-py3
```

3. To use TensorFlow 2, run the following command:
```bash
docker run --gpus all -it -v $(pwd)/dockertmp:/mnt nvcr.io/nvidia/tensorflow:22.01-tf2-py3
```

The command `docker run --gpus all -it -v $(pwd)/dockertmp:/mnt nvcr.io/nvidia/tensorflow:22.01-tf2-py3` is used to run a Docker container with specific configurations and parameters. Let's break down the command:
- `docker run` is the command to run a Docker container.
- `--gpus all` is an argument that allows the container to access all available GPUs on the host machine.
- `-it` is used to allocate an interactive pseudo-TTY, which enables interaction with the container through the command line.
- `-v $(pwd)/dockertmp:/mnt` is an option that creates a volume mount, linking the `dockertmp` directory on the host machine to the `/mnt` directory inside the container. This allows for data sharing between the host and the container.
- `nvcr.io/nvidia/tensorflow:22.01-tf2-py3` is the name and tag of the Docker image used to create the container. In this case, it refers to the NVIDIA CUDA-enabled TensorFlow image with version 22.01, running on Python 3.

Overall, this command launches a Docker container based on the specified TensorFlow image, with access to all GPUs, an interactive terminal, and a volume mount for data sharing between the host and the container.

### Command Output

The output you provided is from running the command `docker run --gpus all -it -v $(pwd)/dockertmp:/mnt nvcr.io/nvidia/tensorflow:22.01-tf2-py3`. Here's the breakdown of the output:
- The header section indicates the TensorFlow version and the NVIDIA release version of the container image. In this case, it is TensorFlow version 2.7.0 and NVIDIA Release 22.01-tf2.
- The container image is provided by NVIDIA Corporation and its affiliates. The copyrights for the image and its contents are mentioned, and the terms and conditions of the NVIDIA Deep Learning Container License apply.
- The output includes a note about the MOFED driver being detected, but the nv_peer_mem driver not being detected. This may affect the multi-node communication performance.
- Another note suggests that the SHMEM (shared memory) allocation limit is set to the default of 64MB, which might be insufficient for TensorFlow. It recommends using additional flags (`--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`) to adjust the container's memory settings.
- Finally, the command prompt `root@e3afd40b59c8:/workspace#` indicates that you are now inside the running container, with the current working directory set to `/workspace`.
- This output confirms that the Docker container has been successfully started using the specified image, and you can now interact with the container's command line for further operations or running TensorFlow applications.

### Container-level Setup vs. Virtual Environment Setup

There are two distinct approaches for working with a running container: Container-level setup and Virtual environment setup. Each method offers its own advantages and considerations.

### Container-level setup

Container-level setup involves directly configuring and installing the necessary dependencies and libraries within the running container itself. This approach allows for a more streamlined and integrated environment, as the container is customized to meet the specific requirements of the project. It eliminates the need for additional virtual environments and simplifies the deployment process. However, it may require administrative privileges and can potentially lead to conflicts if multiple projects with conflicting dependencies are running within the same container.

### Virtual environment setup

On the other hand, Virtual environment setup involves creating isolated virtual environments within the running container. These virtual environments provide a separate workspace with its own set of installed packages and dependencies, ensuring project-specific encapsulation. This approach allows for better management of dependencies, version control, and reproducibility. It also provides flexibility in working with different projects within the same container, as each project can have its own isolated environment. However, it requires additional steps to create and activate the virtual environment and may involve more complex configuration and maintenance.

Choosing between Container-level setup and Virtual environment setup depends on the specific project requirements, organizational preferences, and the complexity of the development environment. Both approaches have their merits, and the decision should be made based on factors such as project scope, scalability, maintainability, and collaboration needs.

## Case Scenarios

The following step-by-step practical examples illustrate the two approaches: Container-level setup and Virtual environment setup.

### Container-level setup

1. Create a new file called `mnist.py` in the workspace directory:
   ```
   root@e3afd40b59c8:/workspace# nano mnist.py
   ```
2. Copy the following code to `mnist.py`:
```
import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Configure TensorFlow to use GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Num Logical GPUs: ", len(logical_gpus))
    except RuntimeError as e:
        print(e)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
```
4. Exit by clicking `ctrl+x` and type `y` to save the file.
5. Execute the script by issuing the following command:
   ```
   root@e3afd40b59c8:/workspace# python mnist.py
   ```
   The output will show the script execution results:
   
  ``` 
   Num GPUs Available:  2
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11501568/11490434 [==============================] - 0s 0us/step
2023-06-20 19:54:56.708815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14635 MB memory:  -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0
Num Logical GPUs:  1
Epoch 1/5
2023-06-20 19:54:59.171541: I tensorflow/stream_executor/cuda/cuda_dnn.cc:377] Loaded cuDNN version 8302
1875/1875 [==============================] - 9s 3ms/step - loss: 0.2185 - accuracy: 0.9385 - val_loss: 0.0945 - val_accuracy: 0.9729
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0763 - accuracy: 0.9779 - val_loss: 0.0596 - val_accuracy: 0.9810
Epoch 3/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0578 - accuracy: 0.9832 - val_loss: 0.0542 - val_accuracy: 0.9811
Epoch 4/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0469 - accuracy: 0.9856 - val_loss: 0.0522 - val_accuracy: 0.9835
Epoch 5/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0394 - accuracy: 0.9881 - val_loss: 0.0525 - val_accuracy: 0.9824
313/313 - 1s - loss: 0.0525 - accuracy: 0.9824

Test accuracy: 0.9824000000953674
```

### Virtual environment setup

1. Install `virtualenv` using pip by running the following command:
   ```
   python3 -m pip install virtualenv
   ```
2. Once `virtualenv` is installed, you can create a virtual environment by running the following command:
   ```
   python3 -m virtualenv myenv
   ```
   Replace `myenv` with the desired name for your virtual environment.
3. Activate the virtual environment using the appropriate command for your shell. For example:
   ```
   source myenv/bin/activate
   ```
   After running this command, you should see the prompt change, indicating that you are now working within the virtual environment.
4. Install the required packages within the virtual environment by running the following command:
   ```
   pip install tensorflow keras wandb scikit-image scipy numpy pandas matplotlib opencv-python pillow pydicom urllib3
   ```
   
   Alternatively, you can save all these packages with their specific versions under `requirements.txt` and install them using the following command:
   ```
   pip install -r requirements.txt
   ```
   
   Here is an example of requirements.txt:
   
``` 
TensorFlow version: 2.12.0
Keras version: 2.12.0
wandb version: 0.15.4
scikit-image version: 0.19.3
scipy version: 1.10.1
numpy version: 1.22.4
pandas version: 1.5.3
matplotlib version: 3.7.1
OpenCV version: 4.7.0
Pillow version: 8.4.0
pydicom version: 2.4.0
urllib3 version: 3.10
```

5. Test the virtual environment with all the available libraries by executing the following code:
```
import tensorflow as tf
import keras
import wandb
import skimage
import scipy
import numpy as np
import pandas as pd
import matplotlib
import cv2
import PIL
import pydicom
import urllib3

# Check GPU support in TensorFlow
print("GPU support in TensorFlow:", tf.test.is_gpu_available())

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("wandb version:", wandb.__version__)
print("scikit-image version:", skimage.__version__)
print("scipy version:", scipy.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("matplotlib version:", matplotlib.__version__)
print("OpenCV version:", cv2.__version__)
print("Pillow version:", PIL.__version__)
print("pydicom version:", pydicom.__version__)
print("urllib3 version:", urllib3.__version__)
```

The output displays the versions of the installed libraries and indicates the availability of two GPUs:

```
(myenv) root@e3afd40b59c8:/workspace# python  versions.py
2023-06-20 20:05:23.119130: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:From versions.py:15: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
2023-06-20 20:05:28.637587: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /device:GPU:0 with 14569 MB memory:  -> device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0
2023-06-20 20:05:28.638372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /device:GPU:1 with 14569 MB memory:  -> device: 1, name: Tesla V100-PCIE-16GB, pci bus id: 0000:86:00.0, compute capability: 7.0
GPU support in TensorFlow: True
TensorFlow version: 2.12.0
Keras version: 2.12.0
wandb version: 0.15.4
scikit-image version: 0.21.0
scipy version: 1.10.1
numpy version: 1.23.5
pandas version: 2.0.2
matplotlib version: 3.7.1
OpenCV version: 4.7.0
Pillow version: 9.5.0
pydicom version: 2.4.0
urllib3 version: 1.26.16
```

### Additional setup

In some cases, additional setup steps may be required. For example, if you encounter OpenGL-related issues, you can update the package lists on the system and install the `libgl1-mesa-glx` package by running the following command:
```
apt-get update && apt-get install -y libgl1-mesa-g
