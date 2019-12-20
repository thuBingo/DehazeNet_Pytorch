import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
from torchvision import transforms
import torch.utils.data as data
#import torchsnooper
import cv2

BATCH_SIZE = 128
EPOCH = 10

# BRelu used for GPU. Need to add that reference in pytorch source file.
class BRelu(nn.Hardtanh):
	def __init__(self, inplace=False):
		super(BRelu, self).__init__(0., 1., inplace)
		
	def extra_repr(self):
		inplace_str = 'inplace=True' if self.inplace else ''
		return inplace_str


class DehazeNet(nn.Module):
	def __init__(self, input=16, groups=4):
		super(DehazeNet, self).__init__()
		self.input = input
		self.groups = groups
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
		self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
		self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
		self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6)
		self.brelu = nn.BReLU()
		for name, m in self.named_modules():
			# lambda : 定义简单的函数    lambda x: 表达式
			# map(func, iter)  iter 依次调用 func
			# any : 有一个是true就返回true
			if isinstance(m, nn.Conv2d):
				# 初始化 weight 和 bias
				nn.init.normal(m.weight, mean=0,std=0.001)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
	
	def Maxout(self, x, groups):
		x = x.reshape(x.shape[0], groups, x.shape[1]//groups, x.shape[2], x.shape[3])
		x, y = torch.max(x, dim=2, keepdim=True)
		out = x.reshape(x.shape[0],-1, x.shape[3], x.shape[4])
		return out
	#BRelu used to CPU. It can't work on GPU.
	def BRelu(self, x):
		x = torch.max(x, torch.zeros(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
		x = torch.min(x, torch.ones(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
		return x
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.Maxout(out, self.groups)
		out1 = self.conv2(out)
		out2 = self.conv3(out)
		out3 = self.conv4(out)
		y = torch.cat((out1,out2,out3), dim=1)
		#print(y.shape[0],y.shape[1],y.shape[2],y.shape[3],)
		y = self.maxpool(y)
		#print(y.shape[0],y.shape[1],y.shape[2],y.shape[3],)
		y = self.conv5(y)
		# y = self.relu(y)
		# y = self.BRelu(y)
		#y = torch.min(y, torch.ones(y.shape[0],y.shape[1],y.shape[2],y.shape[3]))
		y = self.brelu(y)
		y = y.reshape(y.shape[0],-1)
		return y


loader = torchvision.transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
augmentation = torchvision.transforms.Compose([
	transforms.RandomHorizontalFlip(0.5),
	transforms.RandomVerticalFlip(0.5),
	transforms.RandomRotation(30),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FogData(Dataset):
	# root：图像存放地址根路径
	# augment：是否需要图像增强
	def __init__(self, root, labels, augment=True):
		# 初始化 可以定义图片地址 标签 是否变换 变换函数
		self.image_files = root
		self.labels = torch.cuda.FloatTensor(labels)
		self.augment = augment   # 是否需要图像增强
		# self.transform = transform

	def __getitem__(self, index):
		# 读取图像数据并返回
		if self.augment:
			img = Image.open(self.image_files[index])
			img = augmentation(img)
			img = img.cuda()
			return img, self.labels[index]
		else:
			img = Image.open(self.image_files[index])
			img = loader(img)
			img = img.cuda()
			return img, self.labels[index]

	def __len__(self):
		# 返回图像的数量
		return len(self.image_files)


path_train = []
file = open('path_train.txt', mode='r')
content = file.readlines()
for i in range(len(content)):
	path_train.append(content[i][:-1])

label_train = []
file = open('label_train.txt', mode='r')
content = file.readlines()
for i in range(len(content)):
	label_train.append(float(content[i][:-1]))
	#print(float(content[i][:-1]))

train_data = FogData(path_train, label_train, False)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )

net = DehazeNet()
net.load_state_dict(torch.load(r'defog4_noaug.pth', map_location='cpu'))

#@torchsnooper.snoop()
def train():
	lr = 0.00001
	optimizer = torch.optim.Adam(net.parameters(), lr=0.0000005)
	loss_func = nn.MSELoss().cuda()
	for epoch in range(EPOCH):
		total_loss = 0
		for i, (x, y) in enumerate(train_loader):
	# 输入训练数据
	# 清空上一次梯度
			optimizer.zero_grad()
			output = net(x)
	# 计算误差
			loss = loss_func(output, y)
			total_loss = total_loss+loss
	# 误差反向传递
			loss.backward()
	# 优化器参数更新
			optimizer.step()
			if i % 10 == 5:
				print('Epoch', epoch, '|step ', i, 'loss: %.4f' % loss.item(), )
		print('Epoch', epoch, 'total_loss', total_loss.item())
	torch.save(net.state_dict(), r'defog4_noaug.pth')


#train()


def defog(pic_dir):
	img = Image.open(pic_dir)
	img1 = loader(img)
	img2 = transforms.ToTensor()(img)
	c, h, w = img1.shape
	patch_size = 16
	num_w = int(w / patch_size)
	num_h = int(h / patch_size)
	t_list = []
	for i in range(0, num_w):
		for j in range(0, num_h):
			patch = img1[:, 0 + j * patch_size:patch_size + j * patch_size,
				0 + i * patch_size:patch_size + i * patch_size]
			patch = torch.unsqueeze(patch, dim=0)
			t = net(patch)
			t_list.append([i,j,t])
	
	t_list = sorted(t_list, key=lambda t_list:t_list[2])
	a_list = t_list[:len(t_list)//100]
	a0 = 0
	for k in range(0,len(a_list)):
		patch = img2[:, 0 + a_list[k][1] * patch_size:patch_size + a_list[k][1] * patch_size,
				0 + a_list[k][0] * patch_size:patch_size + a_list[k][0] * patch_size]
		a = torch.max(patch)
		if a0 < a.item():
			a0 = a.item()
	for k in range(0,len(t_list)):
		img2[:, 0 + t_list[k][1] * patch_size:patch_size + t_list[k][1] * patch_size,
			0 + t_list[k][0] * patch_size:patch_size + t_list[k][0] * patch_size] = (img2[:,
			0 + t_list[k][1] * patch_size:patch_size + t_list[k][1] * patch_size,
			0 + t_list[k][0] * patch_size:patch_size + t_list[k][0] * patch_size] - a0*(1-t_list[k][2]))/t_list[k][2]
	defog_img = transforms.ToPILImage()(img2)
	defog_img.save('/home/panbing/PycharmProjects/defog/test/test.jpg')


defog('/home/panbing/PycharmProjects/defog/test/fogpic.jpg')
