
[![MLOPs user default predict Github Actions](https://github.com/alonmar/users_classification_default/actions/workflows/main.yml/badge.svg)](https://github.com/alonmar/users_classification_default/actions/workflows/main.yml)

# MLOps user classification 


## Assets in repo

* `requirements.txt`:  [View requirements.txt](https://github.com/alonmar/users_classification_default/blob/main/requirements.txt)
* `users_classification.ipynb`: [View Futbol_Predictions_Export_Model.ipynb](https://github.com/alonmar/users_classification_default/blob/main/users_classification.ipynb) EDA and Export model
* `model_binary_class.dat.gz`: [View model_binary_class.dat.gz](https://github.com/alonmar/users_classification_default/blob/main/model/model_binary_class.dat.gz) Exported Model 
* `app.py`:  [View app.py](https://github.com/alonmar/users_classification_default/blob/main/app.py) Flask api
* `mlib.py`:  [View mlib.py](https://github.com/alonmar/users_classification_default/blob/main/mlib.py) Model Handling Library
* `cli.py`: [View cli.py](https://github.com/alonmar/users_classification_default/blob/main/cli.py) Console predict
* `test_mlib.py`:  [View test_mlib.py](https://github.com/alonmar/users_classification_default/blob/main/test_mlib.py) Unit test
* `utilscli.py`: [View utilscli.py](https://github.com/alonmar/users_classification_default/blob/main/utilscli.py) Utility Belt
* `Dockerfile`: [View Dockerfile](https://github.com/alonmar/users_classification_default/blob/main/Dockerfile) 
* `Makefile`: [View Makefile](https://github.com/alonmar/users_classification_default/blob/main/Makefile) 
* `main.yml`: [View main.yml](https://github.com/alonmar/users_classification_default/blob/main/.github/workflows/main.yml) Github Actionsproyect

## CLI Tools

There are two cli tools.  First, the main `cli.py` is the endpoint that serves out predictions.
To predict the height of an MLB player you use the following: *(if you don't run on Windows delete the backslash "`\`")*

`$ python cli.py --perfil ' [{\"edad\": 25, \"montoSolicitado\": 30000, \"montoOtorgado\": 23000, \"genero\": \"Hombre\", \"quincenal\": 1, \"dependientesEconomicos\": 3, \"nivelEstudio\": \"Universidad\", \"fico\": 569, \"ingresosMensuales\": 15000, \"gastosMensuales\": 4500, \"emailScore\": 0, \"browser\": \"CHROME_MOBILE\", \"NUMTDC_AV\": 0}]'`

![predict-class](https://user-images.githubusercontent.com/36181705/156629731-b75efc9e-ad93-4fed-a6ff-dbebf95c6c14.png)


The second cli tool is `utilscli.py` and this perform model retraining, and could serve as the entry point to do more things.
For example, this version doesn't change the default `model_name`, but you could add that as an option by forking this repo.

`$ python ./utilscli.py retrain --tsize 0.3`

![model-retraining](https://user-images.githubusercontent.com/36181705/156631420-2905a54c-9e2c-4159-8ca5-7e884d040669.png)


## Flask Microservice

The Flask ML Microservice can be run many ways.

### Containerized Flask Microservice Locally

You can run the Flask Microservice as follows with the commmand:

`$ python app.py runserver`

![flask-local](https://user-images.githubusercontent.com/36181705/156631614-038cf778-090c-4ed6-b7dd-4b17b85f5b6a.png)

After run flask to serve a prediction against the application, run the [predict.sh](https://github.com/alonmar/users_classification_default/blob/main/predict.sh).

For the bourne shell: 

`sh ./predict.sh`

For bash:

`bash ./predict.sh`

For windows:

https://stackoverflow.com/questions/26522789/how-to-run-sh-on-windows-command-prompt

```
$ bash ./predict.sh                             
Port: 8080
{
  "prediction": {
    "Type_user": "Moroso", 
    "probability to be Moroso": 0.63, 
    "threshold": 0.25
  }
}
```

### Containerized Flask Microservice

Here is an example of how to build the container and run it locally, this is the contents of [run_docker.sh](https://github.com/alonmar/users_classification_default/blob/main/run_docker.sh)

```
#!/usr/bin/env bash

# Build image
docker build --tag=alonmar/users_classification_default . 

# List docker images
docker image ls

# Run flask app
docker run -p 127.0.0.1:8080:8080 alonmar/users_classification_default
```

### AWS app runner Flask Microservice

With the help of `utilscli.py` you can point to a specific host, the service running in app runner 'https://ceatqt4ut7.us-east-1.awsapprunner.com/predict'

![github-actions](https://user-images.githubusercontent.com/36181705/156633914-f37cd054-0ff0-49a6-98d5-013b8fb4d282.png)


## Github Actions

When you push on main Github detect this [main.yml](https://github.com/alonmar/users_classification_default/blob/main/.github/workflows/main.yml) and automatically build Container via Github Actions

![github-actions](https://user-images.githubusercontent.com/36181705/156632619-96c97b47-f518-49e5-b663-1b7016b0e124.png)


