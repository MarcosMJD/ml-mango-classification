# Setup

## Tools
These are the tools used in this project.

- Anaconda: this is the default ML framework, although in this particular project, it is only used to get the python interpreter (i.e. Anaconda is the python version manager).  
  - https://www.anaconda.com/products/distribution  
- Docker and docker-compose
  - https://www.docker.com/products/docker-desktop/  
- Visual Studio Code.  
  - https://code.visualstudio.com/download  
- Windows 10.  
- git and GitBash. Git for Windows includes GitBash.  
  - https://git-scm.com/downloads  
- Kubectl and Kind
  - https://kind.sigs.k8s.io/docs/user/quick-start/  
  - https://kubernetes.io/docs/tasks/tools/  (note that Docker desktop for Windows already includes Kubectl, so this is not needed on Windows)
- AWS
  - awscli
    - https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html  
  - eksctl
    - https://docs.aws.amazon.com/es_es/eks/latest/userguide/eksctl.html  

It is required python=3.9  
You can use your own IDE, OS and python version manager (e.g. pyenv). 
 
## Install CUDA (only if you have a GPU available)

These are the instructions to install CUDA on Windows. In the links listed, there are also instructions for other OS.  
This should also work with WSL2.  
This project uses cuda toolkit v11.8.0 and cuDNN v8.6.0.  

Update nvidia drivers  
	https://www.nvidia.es/Download/index.aspx?lang=en  
Download and install CUDA Toolkit  
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local  
Direct link to version 11.8.0:    
https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe  

Download and install cuddn sdk (Deep Neural Network library (cuDNN):  
	
Download and extract the zlib package from ZLIB DLL  
http://www.winimage.com/zLibDll/zlib123dllx64.zip  
Note: If using Chrome, the file may not automatically download. If this happens, right-click the link and choose Save link as…. Then, paste the URL into a browser window.  
Extract the library and add the directory path of the downloaded zlibwapi.dll to the environment variable PATH.  
For instance, in any of the directories of cuDNN (see below).  

Download cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x  
https://developer.nvidia.com/cudnn  
https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/  cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip  

Unzip the cuDNN package. Cudnn-windows-x86_64-*-archive.zip. You must replace 8.x and 8.x.y.z with your specific cuDNN version  
Copy the following files from the unzipped package into the NVIDIA cuDNN directory:  
- Copy bin\cudnn*.dll to C:\Program Files\NVIDIA\CUDNN\v8.x\bin.  
- Copy include\cudnn*.h to C:\Program Files\NVIDIA\CUDNN\v8.x\include.  
- Copy lib\cudnn*.lib to C:\Program Files\NVIDIA\CUDNN\v8.x\lib.  
Add Path to PATH environment variables:   
- Open a command prompt from the Start menu.  
- Type Run and hit Enter.  
- Issue the control sysdm.cpl command.  
- Select the Advanced tab at the top of the window.  
- Click Environment Variables at the bottom of the window.  
- Add the NVIDIA cuDNN bin directory path to the PATH variable:  
- Variable Name: PATH   
- Value to Add: C:\Program Files\NVIDIA\CUDNN\v8.x\bin  

## Install Anaconda 
Go to:  
https://www.anaconda.com/products/distribution and install Anaconda for Windows or your OS.  

## Install Kind
https://kubernetes.io/docs/tasks/tools/  
`curl.exe -Lo kind-windows-amd64.exe https://kind.sigs.k8s.io/dl/v0.17.0/kind-windows-amd64`  
`Move-Item .\kind-windows-amd64.exe c:\some-dir-in-your-PATH\kind.exe`  

## Install Kubectl (if you are not on Windows)
https://kubernetes.io/docs/tasks/tools/

## AWS

Only if you wish to deploy the model in EKS

## Install awscli and eksctl
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html  
https://docs.aws.amazon.com/es_es/eks/latest/userguide/eksctl.html  
Note: If you do not want to install eksctl with choco, there is a link to the binary.

## Create an AWS account
- Create an AWS account: https://aws.amazon.com/free/  
- Create an access key (id an secret): https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html    
- Setup the AWS account in your local computer. In a bash, execute:  
```
  aws configure
```
And enter the required information:  
```
  AWS Access Key ID [None]: [your aws key id]
  AWS Secret Access Key [None]: [your asw secret access key]
  Default region name [None]: eu-west-1
  Default output format [None]:
```
Check with `aws sts get-caller-identity`