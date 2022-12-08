# Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations

Paper Link: [arxiv](https://arxiv.org/abs/2207.14083)


## Code
### Requirements
1. git clone --recurse-submodules https://github.com/siyueyu/SCWSSOD.git
2. pip install -r requirements.txt

### Pretrained weights
The pretrained weight can be found here: https://drive.google.com/file/d/1arzcXccUPW1QpvBrAaaBv1CapviBQAJL/view

### Training
1. Download the dataset and pretrained model.
2. Modify the path in tarin.py
3. python train.py

### Testing
The evaluation is done using the submodule [PySODEvalToolKit](https://github.com/lartpang/PySODEvalToolkit.git). Add the json files according to its instruction. Then modify the path and filename, and run `python test.py`.

### Trained model weights and predicted maps

### Credit
The code is based on [SCWSSOD](https://github.com/siyueyu/SCWSSOD.git), [GCPANet](https://github.com/JosephChenHub/GCPANet) and [GatedCRFLoss](https://github.com/LEONOB2014/GatedCRFLoss).