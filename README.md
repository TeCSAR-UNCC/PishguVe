# PishguVe: Attention-utilizing Graph Isomorphism and CNN based Vehicle Trajectory Prediction Architecture


This repository contains official Pytorch impleoemntaion of PishguVe, a novel lightweight vehicle trajectory prediction deep learning architecture that uses attention-based graph isomorphism and convolutional neural networks. 

## Domains and Datasets
- Bird's-eye View: NGSIM Dataset
- Eye-level View: [Carolians Highway dataset](https://github.com/TeCSAR-UNCC/Carolinas_Dataset)
- High-angle View: [Carolians Highway dataset](https://github.com/TeCSAR-UNCC/Carolinas_Dataset)

## Installation 
```bash
git clone https://github.com/TeCSAR-UNCC/PishguVe.git
cd PishguVe
pip install -r requirments.txt
```

## Training and Testing
For training and saving the model in the Training section just set the "save_model" and "train" fields to True in the config file and use the following command:
```
python3 main.py --config {path_to_the_config_file}
```

For testing, just give the path to desired model in the config file and set "save_model" and "train" fields to False and use the same command:
```
python3 main.py --config {path_to_the_config_file}
```

## Citation

If you find our work helpful, please cite the following papers

```
@article{katariya2023pov,
  title={A POV-based Highway Vehicle Trajectory Dataset and Prediction Architecture},
  author={Katariya, Vinit and Noghre, Ghazal Alinezhad and Pazho, Armin Danesh and Tabkhi, Hamed},
  journal={arXiv preprint arXiv:2303.06202},
  year={2023}
}

@article{noghre2022pishgu,
  title={Pishgu: Universal Path Prediction Architecture through Graph Isomorphism and Attentive Convolution},
  author={Noghre, Ghazal Alinezhad and Katariya, Vinit and Pazho, Armin Danesh and Neff, Christopher and Tabkhi, Hamed},
  journal={arXiv preprint arXiv:2210.08057},
  year={2022}
}
```
