import os
import random
import cv2
import numpy as np
import random


def create_dataset(img_dir, data_dir, num_t=10, patch_size=16):
	# img_dir: dir of haze-free images
	# num_t: number of t(x)
	# patch_size: size of image patch
	
	img_path = os.listdir(img_dir)
	
	path_train = []
	label_train = []
	
	for image_name in img_path:
		fullname = os.path.join(img_dir, image_name)
		img = cv2.imread(fullname)
		
		w, h, _ = img.shape
		
		num_w = int(w / patch_size)
		num_h = int(h / patch_size)
		for i in range(1, num_w-1):
			for j in range(1, num_h-1):
				
				free_patch = img[0 + i * patch_size:patch_size + i * patch_size,
					0 + j * patch_size:patch_size + j * patch_size, :]
				
				for k in range(num_t):
					t = random.random()
					hazy_patch = free_patch * t + 255 * (1 - t)
					picname = '%s'%i+'%s'%j+'%s'%k+image_name
					x = random.random()
					if x > 0.5:
						cv2.imwrite(os.path.join(data_dir, picname), hazy_patch)
						path_train.append(os.path.join(data_dir, picname))
						label_train.append(t)
	
	file = open('path_train.txt', mode='a')
	for i in range(len(path_train)):
		file.write(str(path_train[i])+'\n')
	file.close()
	file = open('label_train.txt', mode='a')
	for i in range(len(label_train)):
		file.write(str(label_train[i])+'\n')
	file.close()


create_dataset('/home/panbing/PycharmProjects/defog/img', '/home/panbing/PycharmProjects/defog/dataset', num_t=10, patch_size=16)
