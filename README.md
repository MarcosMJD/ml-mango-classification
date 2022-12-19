# ML for classification of mango varieties
Developing and productization of a Machine Learning model for Classification of Mango Varieties.  
Gamification with client app that challenges the user to compete against the IA.   

### Dataset description
https://www.kaggle.com/datasets/saurabhshahane/mango-varieties-classification  

This dataset contains images of eight varieties of Pakistani mangoes. Automated classification and grading of harvested mangoes will facilitate farmers in delivering high-quality mangoes on time for export, and a high accuracy may be achieved using Convolutional Neural Network.

### Technologies
- Python
- Tensorflow / Tensorflow Lite  
- Keras  
- Models: Xception, EfficientNetB2
- Numpy, Pandas, MatplotLib
- Flask and FastAPI  
- Streamlit  
- Docker / docker-compose  
- Kubernetes / Kind  
- AWS EKS  
- Locust  
- Pytest  

<video src='https://user-images.githubusercontent.com/74185356/208427747-35294c8f-c593-4c33-9b39-1a00deda2803.mp4'></video>

### Development Plan: 
- Setup environment
- Download Dataset
- Development -> notebook.ipynb, train.py, convert_model.py
  - Visualize images
  - Prepare dataset: train-val-test split
  - Create model Xception (imagenet coefs)
	- Add checkpoints
	- Select learning rate
	- Add inner layers
	- Add dropout regularization
	- Perform data augmentation
	- Train with larget files (299x299)
	- Test the model with test set
	- Train efficient-net-b2 for comparison
  - Final model trainning script using the full_train and test datasets
  - Model conversion to tf_lite
- Productization -> ./production/*
  - Create tf-serving image
  - Create gateway (flask) image
  - Create tests for testing the images (with docker-compose)
  - Create Kubernetes yaml files (kind and EKS)
    - gateway service and deployment
    - tf-serving service and deployment
  - Create Streamlit app
  - Performance test with Locuts
- Documentation

### Todo: 
- Documentation (code)  
- Development:  
  - Add L1 and L2 regularization in inner layers  
  - Test with 80-10-10 folds.  
- Production:
  - Automatically get the names of the input/output from the default signature  
  - Find solution to use grpc with asyc/await (FastAPI)  
    - https://docs.python.org/3/library/asyncio-future.html  
    - https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py  
  - Add simple makefile  

### Bugs:
- Check why pytest can non import modules in unit tests of gateway. Meantime, use `python unit_tests.py`  

## Setup

Follow the instructions in [SETUP.md](./SETUP.md)  

### Clone the repo
Open a shell (e.g. Powershell) and execute:  
`git clone https://github.com/MarcosMJD/ml-mango-classification.git`

### Download the dataset
Go to https://www.kaggle.com/datasets/saurabhshahane/mango-varieties-classification  
Create a Kaggle account and click Download button.  
The zip file actually contains two datasets, each one in a subdirectories. We only need the Classification_dataset.   
So unzip the folder `Classification_dataset` into the `data` folder of the repository. Like this:  
/data/Classificacion_dataset/Anwar Ratool  
/data/Classificacion_dataset/Chanusa (Black)  
...

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

To create the development environment and install dependencies, go to the sources/development directory and execute:     
`pipenv install --dev`  
Activate the environment  
`pipenv shell`    

Please, note that the production/gateway and production/fron-end have their specific environments in order to keep these environments separated from the development environment and from each other.  

To check the tensorflow version and GPU devices used (if any):
`python`  
`import tensorflow as tf`  
`tf.config.list_physical_devices('GPU')`  
`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`  

## 1.- Development:

**jupyter notebook**  
In a bash go to `/sources/development` and ensure that the development environment is active, then start jupyter notebook with:
`jupyter notebook`  

Go the browser and open `notebook.ipynb`   
Run all cells check the tranning of models and model evaluation.
The models are saved under the `./models` directory by using checkpoints, which called when the model accuracy improves for each epoch.  
 
Finally, the final model is trained with images of size of 299x299.   

The best model is selected and tested with the test dataset.
Finally, the model is converted into saved_model format under `production\tf-serving` directory by calling the `convert_model.py` script.  
 
Check the model:
Go to `/sources/production/tf-serving/` and run
`saved_model_cli show --dir mango-model --all`

IMPORTANT NOTE:
Before deploying the model, it is needed to set the correct input and outputs in the gateway. To do this:
- Go to the directory `/sources/production/tf-serving`
- Run `saved_model_cli  show --all --dir mango-model`
- Find the line `signature_def['serving_default']:`
- Take the name of the input. E.G. `input_2` in the `line inputs['input_2']`
- the same for the output: `dense_1` in `outputs['dense_1']`
Your numbers may be different.

Edit `/sources/production/gateway/gateway.py` and modify `input_xx` and `dense_xx` with your numbers

**Train script**  
The model with the best parameters will be trained by using the full_train dataset and checking its performance with the test_dataset.  
Is a shell with the development environment activated, run:  
`python train.py`  
The best checkpoint is automatically selected and the model is converted into saved_model format under `production\tf-serving` directory by calling the `convert_model.py` script.  

## 2.- Production:

### Build the images:
Go to /sources/production/tf-serving/ and run:  
`docker build -t tf-serving-mango:v1 .`
Go to /sources/production/gateway/ and run:  
`docker build -t gateway-mango:v1 .`

### Test localy running services in docker containers:

Go to `sources/production/gateway` and activate the environment with: 
`pipenv install --dev`
`pipenv shell`
If you were in the development environment, run `exit` prevously, since nested environments are not permitted.
Go to `/sources/production/integration-tests` and run:
```
docker-compose up
python test_tf_serving.py
python test_gateway.py
CTRL+C
docker-compose down
```
In both cases the results with the prediction for the sample image will be shown. Should be something similar to:  
`{'Anwar Ratool': 1.0, 'Chaunsa (Black)': 0.0, 'Chaunsa (Summer Bahisht)': 0.0, 'Chaunsa (White)': 5.0123284672731474e-37, 'Dosehri': 0
.0, 'Fajri': 2.4020681443702053e-25, 'Langra': 8.949817257817495e-34, 'Sindhri': 5.200761237684644e-35}`  

### Test locally with Streamlit app "Mango game" 

The application 'Mango game' is a Streamlit app that
- Shows a set of sample images as a reference
- Chooses a random image from another set of images
- Asks the user to predict the variety
- Predicts the variety by sending a request to the gateway
- Shows the winner and overall result

In order to run this app:
- In a new bash, Go to `/sources/production/front-end/`
- Run `pipenv install` and `pipenv shell` to activate the environment
- Run `streamlit run client.py`
- Go to the browser at `http://localhost:8501`
- Follow the instructions.

### Create a local Kubernetes cluster with KinD

In a bash, go to `/production/k8s/kind/` and run:
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

In order to access the service (load balancer) in the cluster, we will use port forwarding, since the servicehas not been assined with an IP yet.
```bash
kubectl port-forward service/gateway 8080:80
```
Host port 8080 is forwarding to port 80 of the gateway service where it is listening to. Actually Port 80 of service forwards to port 9000 of container, where gunicorn is listening to.  

Test the gateway service:  
Ensure that the production environment is activated (the one under the production/gateway folder)  
Go to `production/integration-tests/` folder and run  
`python test_gateway.py`  

You may with to delete the cluster with (now, or after the performance tests):  
`./kind delete cluster tf-gateway`  

### Performance tests

For some reason, I have not been able to run Locust within the virtual environment made by pipenv.
I've had to install Locust on the Anaconda environment and launch Locust from Anaconda environment (that is, not from the pipenv environment)
So if needed, use `pip install locust` in your conda environment (you may have already created one for this project)
Go to `sources/production/integration_tests/` and execute `locust -H http://localhost:8080`
Be sure to have a local ML server running, whether docker-compose of with kind as explained before.
Then, go to `http://localhost/8089` and enter 4 users. 


### Deploy to AWS EKS

Please, note that this will have associated costs to the AWS services deployed.  

Update to the latest awscli version and eksctl version.  

`aws ecr create-repository --repository-name mango-repo`

Get the "repository uri", for instance:  
`546106488772.dkr.ecr.eu-west-1.amazonaws.com/mango-repo`  

Tag your local images to point to the ECR repository:  
`docker tag tf-serving-mango:v1 546106488772.dkr.ecr.eu-west-1.amazonaws.com/mango-repo:tf-serving-mango-v1`
`docker tag gateway-mango:v1 546106488772.dkr.ecr.eu-west-1.amazonaws.com/mango-repo:gateway-mango-v1`

Log into ECR with aws cli (change your account id and region accordingly):   
`aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${var.account_id}.dkr.ecr.${var.region}.amazonaws.com`

For instance:
`aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 546106488772.dkr.ecr.eu-west-1.amazonaws.com`  

Push the images:  
`docker push 546106488772.dkr.ecr.eu-west-1.amazonaws.com/mango-repo:tf-serving-mango-v1`
`docker push 546106488772.dkr.ecr.eu-west-1.amazonaws.com/mango-repo:gateway-mango-v1`

Edit gateway-deployment.yaml and tf-serving-deployment.yaml to use these your new tagged images  

`eksctl create cluster -f eks-config.yaml`

kubectl will be contigured to work with eks automatically.  

Apply the configurations:  

```
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml
kubectl apply -f gateway-deployment.yaml
kubectl apply -f gateway-service.yaml
```

Get the load balancer (service) external ip:  
`kubectl get service`
For instance: `a5392f8c5030f410280c2db849511f09-2136527655.eu-west-1.elb.amazonaws.com`
Note: Wait some time for the DNS to propagate. 

You may check the FastAPI by directly opening the API endpoint:  
For instance: `http://a5392f8c5030f410280c2db849511f09-2136527655.eu-west-1.elb.amazonaws.com/docs`  

Or go to /sources/production/front-end and run
`pipenv streamlit client-py`  

Introduce the url of the EKS load balancer followed by `/predict`  
For instance, `http://a5392f8c5030f410280c2db849511f09-2136527655.eu-west-1.elb.amazonaws.com/predict`  

Finally, delete the cluster:
`eksctl delete cluster --name ml-mango-eks`

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
