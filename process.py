import cv2
import logging
import os
import sys
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from skimage.feature import graycomatrix, graycoprops

homeDir = "/home/pitik/Progres/"
pathPic = "hasilCam/"
modelPath = "/home/pitik/Progres/model/cnn_model.h5"
model = load_model(modelPath,compile=False)
model.compile(loss='categorical_crossentropy', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

# untuk glcm
img = []
labels = []
descs = []
# GLCM properties
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
glcm_all_agls = []
columns = []
angles = ['0', '45', '90','135']
kaTelur = ['fertil_10-12', 'fertil_13-15', 'fertil_16-18', 'fertil_4-6', 'fertil_7-9', 'infertil_00']

def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
  glcm = graycomatrix(img,
                      distances=dists,
                      angles=agls,
                      levels=lvl,
                      symmetric=sym,
                      normed=norm)
  feature = []
  glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
  for item in glcm_props:
    feature.append(item)
  return feature

def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)

def cekPic(namePic):
  img = cv2.imread(os.path.join(homeDir, pathPic, namePic))
  kirImg = img[0:1200, 50:1050]#(rowstart:rowend,colstart:colend)
  kanImg = img[0:1200, 1580:2580]#(rowstart:rowend,colstart:colend)
  listPic = [kirImg,kanImg]
  for name in properties :
    for ang in angles:
      columns.append(name + "_" + ang)
  key = [9,9]
  file1 = open(f"{homeDir}log/hasil.log", "a")  # append mode
  for n,i in enumerate(listPic):
    kGray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    kPic = cv2.resize(kGray, (0,0), fx=0.5, fy=0.5)
    kGlcm = calc_glcm_all_agls(kPic, props=properties)
    kX = np.array(kGlcm, dtype=np.float32)
    kX = decimal_scaling(kX.reshape(1,-1))
    kY = model.predict(kX)
    # hasil prediksi one hot encoding
    kYH = np.argmax(kY, axis = 1)
    key[n] = kYH[0]
    file1.write(f"Telur_{n}_{kaTelur[key[n]]}\n")
  file1.close()
  file1.close()
  # mengembalikan hasil prediksi dengan kunci yang sesuai 
  # 0='fertil_10-12', 1='fertil_13-15', 2='fertil_16-18', 3='fertil_4-6', 4='fertil_7-9', 5='infertil_00'
  return key
  