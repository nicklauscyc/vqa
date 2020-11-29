from PIL import Image
import os 
import numpy as np 

path = 'C:/Users/lajja/Documents/Fall2020/ML/vqa/test/test/'
#train_img_len = len(os.listdir(path))

foo = 1
sizes = []

for filename in os.listdir(path):
    full_path = path + filename
    img = Image.open(full_path)
    size = img.size 
    sizes.append(size)

msg = "hello"