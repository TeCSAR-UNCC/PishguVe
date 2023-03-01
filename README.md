# PishguVe: Attention-utilizing graph isomorphism and CNN based vehicle trajectory prediction architecture


This repository contains official Pytorch impleoemntaion of PishguVe, a novel lightweight vehicle trajectory prediction deep learning architecture that uses attention-based graph isomorphism and convolutional neural networks. 

## Domains and Datasets
- Bird's-eye View: NGSIM Dataset
- Eye-level View: [Carolians Highway dataset](https://github.com/TeCSAR-UNCC/Carolinas_Dataset)
- High-angle View: [Carolians Highway dataset](https://github.com/TeCSAR-UNCC/Carolinas_Dataset)

## Installation 
```
git clone https://github.com/TeCSAR-UNCC/PishguVe.git
cd PishguVe
pip install -r requirments.txt
```

## Training and Testing
For training and saving the model in the Training section set the "TRAIN" flag and "SAVE_MODEL" to True in main.py and use the following command:
```
python3 main.py 
```

For testing, give the path to desired model and set the "TRAIN" flag to False in main.py and use the same command:
```
python3 main.py
```
