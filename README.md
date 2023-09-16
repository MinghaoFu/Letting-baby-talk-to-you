
<div align="center">

## [Letting baby talk to you](https://github.com/MinghaoFu/Letting-baby-talk-to-you/)

Minghao Fu, Yu Yao, Haoyue Dai

<p align="center">
    <img src="assets/images/logo.png" alt="logo" width="90%">
</p>

</div>

## Environment

This code is tested on Ubuntu 20.04.4 LTS environment (Python 3.8.4, Pytorch 2.0.1, CUDA 11.8, cuDNN 7.0.5) with NVIDIA A100.

## Dataset
- [BabyChillanto](https://ccc.inaoep.mx/~kargaxxi/)
- [DonateACry](https://github.com/gveres/donateacry-corpus) 

## Get Started

Prepare the environment:
```bash
conda create --name ENV python=3.8 -y
conda activate ENV

git clone https://github.com/MinghaoFu/Letting-baby-talk-to-you.git
cd Letting-baby-talk-to-you

pip install -r requirements.txt
```

Train, evaluate, and test:
```bash
python main.py --seg_len 2 --loss 1*cross_entropy_loss+1*supervised_contrastive_loss --data Mix --seeds 1+2+3+4+5 --reweight --model resnet
```
## App
Try [App](https://github.com/MinghaoFu/Letting-baby-talk-to-you/) for free!

<p align="center">
    <img src="assets/images/record.png" alt="logo" width="90%">
</p>
<p align="center">
    <img src="assets/images/output.png" alt="logo" width="90%">
</p>

## Main Results

## License

This repository is licensed under the terms of the MIT license.

*Your star is my motivation to update, thanks!*
