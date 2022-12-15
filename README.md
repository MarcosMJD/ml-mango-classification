# ML for classification of mango varieties
Developing and productization of a Machine Learning model for Classification of Mango Varieties

## Technologies
- Tensorflow / Tensorflow Lite  
- Keras  
- Flask and FastAPI  
- Streamlit  
- Kubernetes / Kind  
- Docker / docker-compose  
- AWS EKS  

## Plan 
- Setup environment
- Download Dataset
- Development
  - Visualize images
  - Prepare dataset: train-val-test split
  - Create model Xception (imagenet coefs)
	- Add checkpoints
	- Select learning rate
	- Add inner layers
	- Add dropout regularization
	- Perform data augmentation
	- Train with larget files (299x299)
  - Final model trainning script
  - Model conversion to tf_lite
- Deployment
  - Create tf-serving image
  - Create gateway (flask) image
  - Create tests for testing the images (with docker-compose)
  - Create Kubernetes yaml files (kind and EKS)
    - gateway service and deployment
    - tf-serving service and deployment
  - Create Streamlit app


### Todo:  
- Documentation (code and readme)
- Deploy on AWS EKS  
- FastAPI  
  - Check recommendations for deployment in containers.   
- Add L1 and L2 regularization in inner layers
- Try EfficientNetB2
- Find solution to use grpc with asyc/await (FastAPI)  
  - https://docs.python.org/3/library/asyncio-future.html
  - https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py
- Add simple makefile


### Bugs:
- Check why pytest can non import modules in unit tests of gateway. Meantime, use python unit_tests.py

## Setup

These are the tools used in this project.

- Anaconda: this is the default ML framework, although in this particular project, it is only used to get the python interpreter.  
- Visual Studio Code.  
- Windows 10.  
- git and GitBash. Git for Windows includes GitBash.  

You can use your own OS and python version manager. It is required python=3.9
 
### Install CUDA

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
https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip

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

### Install Anaconda 

Go to:
https://www.anaconda.com/products/distribution and install Anaconda form Windows.


### Clone the repo

### Create the environment

Open a shell (GitBash, bash or Powershell) and execute the following instructions:

Note:
If there is an error when starting Powershell and the conda base environment is not activated by default (you should see (base)), try with:  
`powershell -executionpolicy remotesigned`  
or in an already launched powershell: `set-executionpolicy remotesigned`  
To check the policy:  
`get-executionpolicy` (default is restricted)  
You may set the default execution policy at any time.

Create a conda environment (recommended):  
In the root directory of the repository, execute:  
`conda create --name py39 python=3.9`  
`conda activate py39`
Install pip and pipenv  
`pip install -U pip` alternatively `python.exe -m pip install -U pip`  
`pip install pipenv`  

To create the environment and install dependencies, go to the sources/development directory and execute:     
`pipenv install --dev`  
Activate the environment  
`pipenv shell`    

Please, note that the production/gateway and production/fron-end have their specific environments to keep these environments separated from the development environment and from each other.  

To check the tensorflow version and GPU devices used (if any):
`python`  
`import tensorflow as tf`  
`tf.config.list_physical_devices('GPU')`  
`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`  

## FAQ
- ImageDataGEnerator name 'scipy' is not defined
Check that scipy is installed in your environment.
Restart jupyter kernel and try again.

## How to run

### Development:
Launch the development environment as explained previously.  
Run `jupyter notebook`...

The models will be stored unthe the `./models` folder.  
Checkpoints will be called when the model accuracy improves for each epoch.  
Finally, the final model is trained with images of size of 299x299.   

Check the best-model (named best-model.*.h5)  
Convert the best model:  
python convert_model.py ./models/best-model.36_0.975.h5
The model will be converted into saved_model format under `production\tf-serving` directory.  

Check the model:
Go to `/sources/production/tf-serving/` and run
`saved_model_cli show --dir mango-model --all`

### Production:

Build the images:
Go to /sources/production/tf-serving/ and run:  
`docker build -t tf-serving-mango:v1 .`
Go to /sources/production/gateway/ and run:  
`docker build -t gateway-mango:v1 .`

Test localy running services in docker containers:

Got to `sources/production/gateway` and activate the environment. If you were in the development environment, run `exit` prevously, since nested environments are not permitted.
Go to `/sources/production/tests` and run:
	docker-compose up
	python test_tf_serving.py
	python test_gateway.py
	docker-compose down

In both cases the results with the prediction for the sample image will be shown. Should be something similar to:  
`{'Anwar Ratool': 1.0, 'Chaunsa (Black)': 0.0, 'Chaunsa (Summer Bahisht)': 0.0, 'Chaunsa (White)': 5.0123284672731474e-37, 'Dosehri': 0
.0, 'Fajri': 2.4020681443702053e-25, 'Langra': 8.949817257817495e-34, 'Sindhri': 5.200761237684644e-35}`  

	
`./kind create cluster --name tf-gateway`
Check with: 
`kubectl cluster-info --context kind-tf-gateway`

Load images to the cluster:
`./kind load docker-image tf-serving-mango:v1 gateway-mango:v1 --name tf-gateway`

Create the deployments and services:  
```
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml
kubectl apply -f gateway-deployment.yaml
kubectl apply -f gateway-service.yaml
```

**Test gateway and its connection with tf-serving**  
```bash
kubectl port-forward service/gateway 9000:80
```
Host port 9000 is forwarding to port 80 of the gateway service where it is listening to. Actually Port 80 of service forwards to port 9000 of container, where gunicorn is listening to.  

Test the gateway service:  
Ensure that the production environment is activated (the one under the production/gateway folder)  
Go to `production/tests/` folder and run  
`python test_gateway.py`  

## Performance tests

For some reason, I have not been able to run Locust within the virtual environment made by pipenv.
I've had to install Locust on the Anaconda environment and launch Locust from Anaconda environment (that is, not from the pipenv environment)
So if needed, use `pip install locust` in your conda environment (you may have already created one for this project)
Go to `sources/production/integration_tests/` and execute `locust -H http://localhost:8080`
Be sure to have a local ML server running, whether docker-compose of with kind as explained before.
Then, go to `http://localhost/8089` and enter 4 users. 

## Useful snippets

**Change kubectl context (cluster)**  
kubectl config use-context kind-<cluster-name>  

**Get clusters**  
./kind get clusters  

**Check images loaded in a node**  
kubectl get nodes  
Note: There should be one which is the control plane  
```
winpty docker exec -ti <<cluster-name>-control-plane> bash
crictl  images
```
**Delete cluster** 
kind delete cluster --name <name>  

**Log in (bash) into a running container**  
`winpty docker exec -it <container-id> bash`

**Load and send image via post with requests**
```
import requests
url = 'http://..."
files = {'media': open('test.jpg', 'rb')}
requests.post(url, files=files)
```

**Get image in Flask via POST**
```
imagefile = flask.request.files.get('imagefile', '')
```

**Convert image file from Flask to PIL**
`pil_image = Image.open(image)`

**To get the class of the highest prediction**  
`np.array(classes)[np.argmax(preds,1)]`  

**To evaluate a model with a dataset**
X can be a keras iterator obtained from keras image data generator -> from dataframe or directory. It returns loss and metric defined in the model compilation.
`model.evaluate(X)`

**Show debug messages with docker-compose**
`print('debug', flush=True)`

**Delete a commit**
git reset --hard <commit-id>

**Delete a file in the last commit**
git reset --soft HEAD~1
(the files will be in the staged area where they can be removed or unstaged)
git rm --cached <file>

**Install Gunicorn and Uvicorn for multiprocess with FastAPI**
`pip install "uvicorn[standard]" gunicorn`

**Run Gunicorn within a docker container with several Uvicorn workers (Linux only)**
```
ENTRYPOINT [ "gunicorn", "gateway:app", "--workers=4", "--worker-class=uvicorn.workers.UvicornWorker",   "--bind=0.0.0.0:9000"]
```

**Run uvicorn for testing FastAPI**  
`uvicorn gateway_template:app --reload`  
`uvicorn gateway:app --reload --host 127.0.0.1 --port 8080`    
On Windows 0.0.0.0 will launch the server, but clients using 0.0.0.0 or 127.0.0.1 or localhost will not work.  
On Windows, port 9000 is denied (permissions).  

**Send image with curl in multipart form data format**
 `curl -X 'POST'   'http://localhost:8000/predict'   -H 'accept: application/json'   -H 'Content-Type: multipart/form-data'   -F 'file=@IMG_20210630_102920.jpg;type=image/jpeg'`

**Get the key with the max value in a dict**
`print(max(x, key=x.get))`

**Example of simple makefile**
```
deps:
	poetry install

dev: deps
	cd app && poetry run python main.py

lint: lint-black lint-flake lint-pycodestyle

lint-black:
	poetry run black --line-length 100 app

lint-flake:
	poetry run flake8 --max-line-length 100 app

lint-pycodestyle:
	poetry run pycodestyle --max-line-length 100 ./app
```