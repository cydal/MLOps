

### https://github.com/kubeflow/pipelines/tree/master/samples


## Mac OS 
## set up Kind

> brew install kind



### Start Minikube
> minikube start


## Deploy up kubeflow pipelines

> # env/platform-agnostic-pns hasn't been publically released, so you will install it from master
export PIPELINE_VERSION=1.7.1
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"


# pipeline ui
> kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

Access UI in localhost/8080

https://www.kubeflow.org/docs/components/pipelines/installation/localcluster-deployment/


## Install pytorch-kfp-components
> pip3 install -U pytorch-kfp-components


## Generate yaml files from templates
> python utils/generate_templates.py cifar10/template_mapping.json


## Generate pipeline yaml file
> python cifar10/pipeline.py


# Upload run



## Run 1 
* Model - Resnet50 
* Pretrained - Yes/ImageNet
* Epoch - 1
* Augmentation - No

### Confusion Matrix
[resnet.jpg](https://postimg.cc/nXQ6J4F3)

### Loss Curve
[Screenshot-2022-01-03-at-00-16-20.png](https://postimg.cc/vg1nqZLx)


# Run 2
* Model - Resnet18
* Pretrained - Yes/ImageNet
* Epoch - 10
* Augmentation - No

### Confusion Matrix
[Screenshot-2022-01-03-at-00-21-08.png](https://postimg.cc/p9vxcGff)


### Loss Curve
[Screenshot-2022-01-03-at-00-29-32.png](https://postimg.cc/dhtjRvXr)


# Run 3
* Model - Image Augmentation 
* Pretrained - Yes/ImageNet
* Epoch - 10
* Augmentation - Yes

### Confusion Matrix
[![Screenshot-2022-01-03-at-00-37-51.png](https://i.postimg.cc/hGYfScrw/Screenshot-2022-01-03-at-00-37-51.png)](https://postimg.cc/PLmtyG0z)

### Loss Curve
[![Screenshot-2022-01-03-at-00-35-40.png](https://i.postimg.cc/Dw2hHHqX/Screenshot-2022-01-03-at-00-35-40.png)](https://postimg.cc/XpDty21j)
