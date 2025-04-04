from pypiqe import piqe
import os
import cv2
import imquality.brisque as brisque
from skimage import io,img_as_float
import torch
import os
import pickle
import typing
import warnings
from enum import Enum

import PIL.Image
import numpy
import scipy.signal
import skimage.color
import skimage.transform
from libsvm import svmutil

# Specify the directory path
# directory_path = r'D:\Pytorch\NTNU\dusgan_modified\results_HAT'
directory_path=r'D:\Pytorch\NTNU\dusgan_modified\ZSSR_results'
file_path1 = r'D:\Pytorch\NTNU\dusgan_modified\results_ZSSR.txt'
# rootdir =r'G:\caps_data\Train'
i=1
# def calculate_brisque(img):
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
#     # img = img_as_float(io.imread(img))
#     # print(img.shape)
#         score = brisque.score(img)
#     return score
# Iterate through all the files in the directory
average_piqe=0
average_brisque=0

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path,filename)
    print(file_path)
    image = cv2.imread(file_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgYCC= imgYCC[:,:,0]
    imgYCC=img_as_float(imgYCC)
    # print(imgYCC)
    # img=img_as_float(img)
    # print(img)
    # # img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # img1=torch.tensor(img1)
    # # img1=torch.unsqueeze(img1,0)   
    # # score1=brisque.score(img)
    score, activityMask, noticeableArtifactMask, noiseMask = piqe(img)
    average_piqe+= score
    # # print(score)
    # # # print('for image number' +str(i) +'piqe score is'+str(i), score)
    # # # print('for image number' +str(i) +'brisque score is'+str(i), score1)
    # # print('-------------')
    # img1=img_as_float(io.imread(file_path))
    # # print(img1)
    # #test=Brisque(img1)
    score1=brisque.score(imgYCC)
    average_brisque+=score1
    # print('for image number' +str(i) +'brisque score is'+str(i), score1)
    with open(file_path1, 'a') as file:
        file.write('for image '+str(i)+' Brisque is'+str(score1)+' PIQE is'+str(score)+'\n')

    i=i+1

average_brisque=average_brisque/i
average_piqe=average_piqe/i
with open(file_path1, 'a') as file:
        file.write('average brisque is '+str(average_brisque)+'  average piqe is  '+str(average_piqe))


#  for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
#     # img = img_as_float(io.imread(img))
#     # print(img.shape)
#         score = brisque.score(img)
#     return score