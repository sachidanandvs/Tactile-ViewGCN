# Tactile-ViewGCN

## Introduction
This is a Pytorch based code for object classification methods presented in the paper "Tactile-ViewGCN: Learning Shape Descriptor from Tactile Data using Graph Convolutional Network".

## System requirements

Requires CUDA and Python 3.6+ with following packages (exact version may not be necessary):

* numpy (1.15.4)
* torch (1.8.0)
* torchvision (0.9.0)
* scipy (1.5.2)
* scikit-learn (0.23.2)

## Dataset preparation

1. Download the `classification` and/or `weights` dataset from [http://humangrasp.io](http://humangrasp.io) .
2. Extract the dataset metadata.mat files to a sub-folder `data\[task]`. The resulting structure should be something like this:
```
data
|--classification
|    |--metadata.mat
|--weights
        |--metadata.mat
```
More information about the dataset structure is availble in [http://humangrasp.io](http://humangrasp.io) .

## Object classification

Run the code from the root working directory (the one containing this readme).

### Training
You can train a model from scratch using:
```
python classification/train.py
```

## Terms
Usage of this dataset (including all data, models, and code) is subject to the associated license, found in [LICENSE](http://humangrasp.io/license.html). The license permits the use of released code, dataset and models for research purposes only.

We also ask that you cite the associated paper if you make use of this dataset; following is the BibTeX entry:

```
@article{
	SSundaram:2019:STAG,
	author = {Sundaram, Subramanian and Kellnhofer, Petr and Li, Yunzhu and Zhu, Jun-Yan and Torralba, Antonio and Matusik, Wojciech},
	title = {Learning the signatures of the human grasp using a scalable tactile glove},
	journal={Nature},
	volume={569},
	number={7758},
	year={2019},
	publisher={Nature Publishing Group}
	doi = {10.1038/s41586-019-1234-z}
}
```