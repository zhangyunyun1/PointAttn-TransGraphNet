# PointAttn-TransGraphNet

This code repository contains the source code for point cloud generation from the following papers:  
**Article Title**: High-Fidelity Point Cloud Generation via Multi-scale Attentive Feature Fusion with GraphSAGE and Transformer

# Datasets

We follow [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet) and evaluate the point cloud generation quality of PointAttn-TransGraphNet on the ShapeNet dataset. Here is the download link for the data:  
- [ShapeNet](https://drive.google.com/drive/folders/1SRJdYDkVDU9Li5oNFVPOutJzbrW7KQ-b)

## Getting Started

### Setting up the environment
```bash
$ cd PointAttn-TransGraphNet
$ conda create -n ptg python=3.7
$ conda activate ptg
# pytorch
$ pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip3 install -r requirements.txt

```
### Building PyTorch Extensions
```bash
cd models/pointnet2_ops_lib
python setup.py install
cd ../..
cd loss_functions/Chamfer3D
python setup.py install
cd ../emd
python setup.py install
```
### Training
To train a point cloud generation model, please run the following commands:
```bash
python train.py
```
### Test
To evaluate the pre-trained model, first specify the checkpoint file save path, then run the following command:
```bash
python test.py
```
## Visual point cloud generation
To generate visual images of chairs and airplanes, please run the following command:
```bash
python train_chair.py
python train_airplane.py
```
## Acknowledgement
Our code is inspired by [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet). We thank the authors for their great job!
