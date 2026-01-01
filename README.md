# CA_segment





## Environment Setup

Create environment and install all dependencies via:

```bash
conda create -n cas_env python=3.11.5
conda activate cas_env
cd CA_sgement
pip install -r requirements.txt # (Not Applicable) To-be Updated
pip install -e .
```

## Dataset
We used **COCA (Coronary Calcium and chest CT’s)** dataset for training and validation. The dataset is available at [LINK](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa). We only use gated coronary CT DICOM images with corresponding coronary artery calcium segmentations and scores. (**EXCEPT non-gated chest CT DICOM images**)

### Preprocessing - (Continually Updated)
At first, you run for generating metadata of COCA dataset. This script will generate **table** including a list of DICOM files and their corresponding metadata and **json files** including (x,y) coordinates of coronary artery calcium segmentations.

First, you should set up your config file `config/config_template.yaml` and change it to `config/config.yaml`. Then, you can run the following command:

```bash
python dataset/COCA/preprocess.py --saveFileName <FILENAME>
```


## Dependencies
- python==3.11.5
- torch==2.9.1
- CUDA==12.6
- monai==1.15.1
- wandb==0.21.2
- opencv-python==4.12.0