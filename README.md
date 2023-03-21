# code for ntire2023 image stereo super resolution team McSR

# Requirements
PyTorch1.13.1，torchvision0.14.1. The code is tested with python=3.9.16 and cuda=11.7.

# Train
## 1. Prepare training data 

├── NTIRE2023Test
│   └── LR_x4_2
│       ├── 0001_L.png
│       ├── 0001_R.png
│       ├── 0002_L.png
│       └── 0002_R.png
├── Train
│   ├── HR
│   │   ├── 0001_L.png
│   │   ├── 0001_R.png
│   │   ├── 0002_L.png
│   │   └── 0002_R.png
│   └── LR_x4
│       ├── 0001_L.png
│       ├── 0001_R.png
│       ├── 0002_L.png
│       └── 0002_R.png
└── Validation
    ├── HR
    │   ├── 0001_L.png
    │   ├── 0001_R.png
    │   ├── 0002_L.png
    │   └── 0002_R.png
    └── LR_x4
        ├── 0001_L.png
        ├── 0001_R.png
        ├── 0002_L.png
        └── 0002_R.png
        
## 2. Begin to train
Run  ```python train.py ``` to perform training. The checkpoint will be saved to ./log/

# Test
## 1. Prepare test data 
Download the test sets from NTIRE 2023 Stereo Image Super Resolution Challenge and unzip them to ./data/NTIRE2023Test

## 2. Begin to test
Run inference.py to perform inference.

## Module Mean:
Run mean_model_weight.py
## Model Ensemble:
Run inference_ensemble.py

## Download the Results:
We share the results achieved by our SwinFSR on NTIRE 2023 Stereo Image Super Resolution test sets for 4xSR. The final submission results are available on Google Drive: https://drive.google.com/file/d/10fWVV2HFl6GBJQs0BSYHRXZfXydPQXam/view?usp=sharing
The result are coming from ensembling the inference results of top 3 models   (perceptual_loss, slowftcam, scam). 
These models and inferences results can be download here: https://drive.google.com/drive/folders/10N-ctXAUbJ4bfQJZmMJDSSa9odL3fj_S?usp=sharing. 
Download these 3 inference results in ./results folder and run inference_ensemble.py to get the final results in ./results_final folder.
