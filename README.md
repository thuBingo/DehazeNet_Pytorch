# DehazeNet_Pytorch
A Pytorch implementation for DehazeNet in paper 'DehazeNet: An End-to-End System for Single Image Haze Removal'

@article{cai2016dehazenet,/n
	author = {Bolun Cai, Xiangmin Xu, Kui Jia, Chunmei Qing and Dacheng Tao},/n
	title={DehazeNet: An End-to-End System for Single Image Haze Removal},/n
	journal={IEEE Transactions on Image Processing},/n
	year={2016}, /n
	volume={25}, /n
	number={11}, /n
	pages={5187-5198},/n
	}
  
Run create_dataset.py to create a training dataset. /n
Run DehazeNet-pytorch.py.train() to train a model. /n
Run DehazeNet-pytorch.py.defog() to defog a picture. /n
The model is trained on GPU and defog() is run on CPU./n
