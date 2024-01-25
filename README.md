## Overview
The FPGA/GPU cluster is a cloud-based, remotely accessible computing infrastructure specifically designed to accelerate compute-intensive applications, such as machine learning training and inference, video processing, financial computing, database analytics networking, and bioinformatics. Latest state-of-the-art acceleration technologies including the Alveo FPGAs, and Tesla V100 GPUs, closely coupled with server processors constitute the backbone of this cluster. The software stack consists of a complete ecosystem of machine learning frameworks, libraries, and runtime targeting heterogeneous computing accelerators.
![image](https://user-images.githubusercontent.com/9284879/119378393-3c6aaf00-bc8c-11eb-92df-92df33ddc13d.png)

## Machine Learning Software Stack
The FPGA/GPU cluster supports three the most commonly used deep learning frameworks, namely, TensorFlow, Caffe, and Pytorch. These frameworks provide a high-level abstraction layer for deep learning architecture specification, model training, tuning, testing, and validation. The software stack also includes various machine learning vendor-specific libraries that provide dedicated computing functions tuned for specific hardware architecture, delivering the best possible performance/power figure.

![image](https://user-images.githubusercontent.com/9284879/119378658-7b990000-bc8c-11eb-8fb0-9ab17804d898.png)

## Remote Access
This quick guide provides instructions on how to reserve, access, manage, and use the FPGA/GPU cluster nodes through the CMC cloud environment interface.
https://www.cmc.ca/qsg-fpga-gpu-cluster/

## Maximizing FPGA/GPU Cluster Potential: A Triad of Development Modes

To fully harness the capabilities of the FPGA/GPU cluster, users can explore three distinct modes, each tailored to different development preferences and requirements: 

-   [Working with Virtual Environments](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/Working%20with%20Virtual%20Environments.md)
Firstly, for an isolated and customizable development environment, utilizing virtual environments is recommended. Detailed guidance on this approach can be found in the documentation titled Working with Virtual Environments. 

-   [Working with Docker](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/Working%20with%20Docker.md)
Alternatively, users keen on containerization practices can benefit from Docker integration, streamlining deployment, and scalability. Refer to Working with Docker for a comprehensive guide on optimizing workflows with Docker. 

-   [Leveraging GPU Power through Remote Development in Visual Studio Code](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/remote%20development%20using%20vscode.md)
Lastly, those preferring the popular Visual Studio Code environment can leverage GPU power remotely for coding, debugging, and machine learning tasks through the guidance provided in Leveraging GPU Power through Remote Development in Visual Studio Code. By exploring these modes, users can tailor their experience to match their specific development needs within the FPGA/GPU cluster infrastructure.
