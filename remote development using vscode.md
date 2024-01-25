
# Overview: Leveraging GPU Power through Remote Development in Visual Studio Code

Remote development with Visual Studio Code (VSCode) empowers users to tap into the immense computational capabilities of GPU resources located on remote machines, all while benefiting from the familiar and feature-rich environment that VSCode provides on their local workstations. This setup is especially advantageous for tasks such as coding, debugging, and training machine learning models that demand substantial computational prowess.

## Installation Process
Getting started is a seamless process. Follow these steps to set up your environment:

- Install Visual Studio Code: Download and install Visual Studio Code.
Install Remote-SSH Extension:
- Enhance VSCode functionality by installing the Remote-SSH extension.
Optional: Install Remote Development Extension Pack:

If collaboration with other remote extensions within VSCode is anticipated, consider installing the Remote Development extension pack.

## Step-by-Step Configuration for GPU Node Access

### Step 1: GPU Node Reservation

1.  Utilize the quick guide available at https://www.cmc.ca/qsg-fpga-gpu-cluster/ to reserve and access a GPU node.
2.  Refer to the example below to configure VSCode for accessing the reserved GPU node:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/8.png?raw=true)

In the example above, you can ssh to the node using the following command:
```bash
ssh yourregisteredcmcemail@uwaccel.cmc.ca -p 49872
```

Where yourregisteredcmcemail is your user name and 49872 is the port number of the reserved node allocated to the ssh trafic.

## Step 2: Configure VS Code for GPU Node Access

1.  Upon successful GPU node reservation and SSH access, open VSCode.
2.  Click on the SSH icon at the bottom left of the VSCode window.

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/1.png?raw=true)

3.  Select "Connect to Host" from the dropdown menu:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/2.png?raw=true)

4.  Choose "Configure SSH Hosts":

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/3.png?raw=true)

5.  Select your local SSH configuration file to update:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/4.png?raw=true)

6.  Edit the file by adding information for the reserved node (e.g., Host, HostName, User, Port):

```bash
Host GPUNode
  HostName uwaccel.cmc.ca
  User yassine.hariri@cmc.ca
  Port 49872
```
Save and close the file C:\Users\yassine\.ssh\config

7. Click again on the SSH icon at the bottom left of the VSCode window:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/1.png?raw=true)

Notice the appearance of "GPUNode" in the dropdown menu of SSH hosts:

8.  Click on the "GPUNode" host. This will connect you to the reserved GPU node:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/7.png?raw=true)

9.  Enter your password:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/5.png?raw=true)

Congratulations! You now have access to the reserved GPU node. Begin your machine learning development on the server:

![image](https://github.com/cmcmicrosystems/FPGA-GPU-Cluster/blob/main/images/6.png?raw=true)

The Remote VSCode session is structured into three integral components. Firstly, a sophisticated code editor tailored to your language of choice offers a seamless coding experience. Secondly, a dedicated terminal window facilitates tasks such as repository cloning, Anaconda virtual environment activation, and the execution of accelerated GPU-based training. The third component is the Remote Explorer, providing remote access to your files and directories, enhancing overall flexibility and productivity in managing your development environment.

By following these comprehensive steps, you've seamlessly configured VSCode for remote development with GPU resources, unlocking the potential for high-performance tasks such as machine learning development.