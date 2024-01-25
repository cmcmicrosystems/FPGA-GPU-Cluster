# Working with Virtual Environments

Anaconda is an open-source platform designed to simplify package management and deployment for data science and machine learning projects. With its Conda package manager, Python interpreter, and a rich set of pre-installed libraries, Anaconda provides a seamless experience for creating isolated environments, managing dependencies, and ensuring cross-platform consistency.

## Key Features

### Package Management

Anaconda utilizes Conda for efficient package management. This makes it easy to install, update, and remove packages, ensuring that your project dependencies are well-maintained.

### Virtual Environments

Anaconda allows you to create isolated environments, enabling you to work with different versions of Python and packages for various projects. This is particularly useful to avoid conflicts and maintain project-specific configurations.

### Cross-Platform Support

Anaconda supports Windows, macOS, and Linux, providing a consistent environment across different operating systems. This feature is crucial for collaborative projects involving team members with diverse setups.

### Data Science Libraries

Anaconda comes with a comprehensive set of pre-installed data science and machine learning libraries, streamlining the setup process for your projects.

## Tensorflow

### Instructions to Activate/Use the Tensorflow Environment

```bash
# Activate the base environment
source /CMC/accelerator/anaconda3/bin/activate

# Switch to the Tensorflow environment
conda activate /CMC/accelerator/anaconda3/envs/tensorflow

# Download and execute a Tensorflow example (e.g., mnist_tf.py)
wget https://raw.githubusercontent.com/cmcmicrosystems/FPGA-GPU-Cluster/main/mnist_tf.py
python mnist_tf.py
```

### Tensorflow Environment Execution Result
```bash
(base) yassine@cmc@uwaccel02:~/mydata$
(base) yassine@cmc@uwaccel02:~/mydata$ conda activate /CMC/accelerator/anaconda3/envs/tensorflow
(tensorflow) yassine@cmc@uwaccel02:~/mydata$ wget https://raw.githubusercontent.com/cmcmicrosystems/FPGA-GPU-Cluster/main/mnist_tf.py
--2024-01-24 14:33:00--  https://raw.githubusercontent.com/cmcmicrosystems/FPGA-GPU-Cluster/main/mnist_tf.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.111.133, 185.199.110.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1434 (1.4K) [text/plain]
Saving to: ‘mnist_tf.py’

mnist_tf.py                                      100%[=========================================================================================================>]   1.40K  --.-KB/s    in 0s

2024-01-24 14:33:00 (15.1 MB/s) - ‘mnist_tf.py’ saved [1434/1434]

(tensorflow) yassine@cmc@uwaccel02:~/mydata$ python mnist_tf.py
2024-01-24 14:33:05.761799: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2024-01-24 14:33:18.196904: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2024-01-24 14:33:18.200826: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2024-01-24 14:33:18.313425: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:3b:00.0 name: Tesla V100-PCIE-16GB computeCapability: 7.0
coreClock: 1.38GHz coreCount: 80 deviceMemorySize: 15.77GiB deviceMemoryBandwidth: 836.37GiB/s
2024-01-24 14:33:18.313747: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 1 with properties:
pciBusID: 0000:86:00.0 name: Tesla V100-PCIE-16GB computeCapability: 7.0
coreClock: 1.38GHz coreCount: 80 deviceMemorySize: 15.77GiB deviceMemoryBandwidth: 836.37GiB/s
2024-01-24 14:33:18.313782: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2024-01-24 14:33:18.407366: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2024-01-24 14:33:18.407485: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.10
2024-01-24 14:33:18.468335: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2024-01-24 14:33:18.579619: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2024-01-24 14:33:18.636549: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2024-01-24 14:33:18.684137: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.10
2024-01-24 14:33:18.847335: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.7
2024-01-24 14:33:18.848959: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0, 1
Num GPUs Available:  2
2024-01-24 14:33:19.331476: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-01-24 14:33:19.336993: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:3b:00.0 name: Tesla V100-PCIE-16GB computeCapability: 7.0
coreClock: 1.38GHz coreCount: 80 deviceMemorySize: 15.77GiB deviceMemoryBandwidth: 836.37GiB/s
2024-01-24 14:33:19.337080: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2024-01-24 14:33:19.337150: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2024-01-24 14:33:19.337189: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.10
2024-01-24 14:33:19.337226: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2024-01-24 14:33:19.337263: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2024-01-24 14:33:19.337299: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2024-01-24 14:33:19.337335: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.10
2024-01-24 14:33:19.337372: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.7
2024-01-24 14:33:19.338328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2024-01-24 14:33:19.338416: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2024-01-24 14:33:20.771725: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2024-01-24 14:33:20.771775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0
2024-01-24 14:33:20.771789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N
2024-01-24 14:33:20.774572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14753 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0)
2024-01-24 14:33:20.775063: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
Num Logical GPUs:  1
2024-01-24 14:33:21.365565: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2024-01-24 14:33:21.383612: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2100000000 Hz
Epoch 1/5
2024-01-24 14:33:21.740512: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2024-01-24 14:33:22.060424: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.7
2024-01-24 14:33:24.197228: W tensorflow/stream_executor/gpu/asm_compiler.cc:63] Running ptxas --version returned 256
2024-01-24 14:33:24.323171: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] Internal: ptxas exited with non-zero error code 256, output:
Relying on driver to perform ptx compilation.
Modify $PATH to customize ptxas location.
This message will be only logged once.
1875/1875 [==============================] - 14s 4ms/step - loss: 0.4213 - accuracy: 0.8779 - val_loss: 0.0941 - val_accuracy: 0.9732
Epoch 2/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0905 - accuracy: 0.9738 - val_loss: 0.0699 - val_accuracy: 0.9774
Epoch 3/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0631 - accuracy: 0.9815 - val_loss: 0.0586 - val_accuracy: 0.9810
Epoch 4/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0502 - accuracy: 0.9856 - val_loss: 0.0607 - val_accuracy: 0.9802
Epoch 5/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0424 - accuracy: 0.9872 - val_loss: 0.0496 - val_accuracy: 0.9837
313/313 - 0s - loss: 0.0496 - accuracy: 0.9837
Test accuracy: 0.9836999773979187
deactivate the environement:
conda deactivate
```

## Pytorch

### Instructions to Activate/Use the Pytorch Environment

```bash
# Activate the base environment
source /CMC/accelerator/anaconda3/bin/activate

# Switch to the Pytorch environment
conda activate /CMC/accelerator/anaconda3/envs/torch

# Download and execute a Pytorch example (e.g., mnist_pytorch.py)
wget https://raw.githubusercontent.com/cmcmicrosystems/FPGA-GPU-Cluster/main/mnist_pytorch.py
python mnist_pytorch.py
```

### Pytorch Environment Execution Result
```bash
(base) yassine@cmc@uwaccel02:~/mydata$ conda activate /CMC/accelerator/anaconda3/envs/torch
(torch) yassine@cmc@uwaccel02:~/mydata$ wget https://raw.githubusercontent.com/cmcmicrosystems/FPGA-GPU-Cluster/main/mnist_pytorch.py
--2024-01-24 14:36:58--  https://raw.githubusercontent.com/cmcmicrosystems/FPGA-GPU-Cluster/main/mnist_pytorch.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.110.133, 185.199.111.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2281 (2.2K) [text/plain]
Saving to: ‘mnist_pytorch.py’

mnist_pytorch.py                                 100%[=========================================================================================================>]   2.23K  --.-KB/s    in 0s

2024-01-24 14:36:59 (24.6 MB/s) - ‘mnist_pytorch.py’ saved [2281/2281]

(torch) yassine@cmc@uwaccel02:~/mydata$ python mnist_pytorch.py

(torch) yassine@cmc@uwaccel02:~/mydata$ python mnist_pytorch.py
Num GPUs Available:  2
Epoch 1/5, Validation Accuracy: 95.73%
Epoch 2/5, Validation Accuracy: 96.77%
Epoch 3/5, Validation Accuracy: 97.55%
Epoch 4/5, Validation Accuracy: 97.38%
Epoch 5/5, Validation Accuracy: 97.58%
Test accuracy: 97.51%
```