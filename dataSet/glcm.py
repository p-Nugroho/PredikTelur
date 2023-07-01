import numpy as np
import cv2
import os
import re
import pandas as pd 
from skimage.feature import graycomatrix, graycoprops
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
datasetDir = "aaDataSet"
imgs = [] #list image matrix
labels = []
descs = []
# GLCM properties
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
glcm_all_agls = []
columns = []
angles = ['0', '45', '90','135']

def normalize_label(folder, str_):
  str_ = str_.replace("Day", "")
  intStr = int(str_)
  if folder == "fertil":
    if intStr >= 4 and intStr <= 6:
      str_ = folder + "_4-6" 
    if intStr >= 7 and intStr <= 9:
      str_ = folder + "_7-9"
    if intStr >= 10 and intStr <= 12:
      str_ = folder + "_10-12"
    if intStr >= 13 and intStr <= 15:
      str_ = folder + "_13-15"  
    if intStr >= 16 and intStr <= 18:
      str_ = folder + "_16-18" 
  if folder == "infertil":
    str_ = folder + "_00"
  return str_

# def normalize_desc(folder, str_):
#   text = folder + "_" + str_ 
#   return text

def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
  progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
  if val == 0:
    print("", end = "\n")
  else:
    print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")


# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
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
  feature.append(label) 
  
  return feature

for folder in os.listdir(datasetDir):
  # print(folder) #folder ferttil dan inferrtil
  for sub_folder in os.listdir(os.path.join(datasetDir, folder)):
    # print(sub_folder) #folder dayxx
    sub_folder_files = os.listdir(os.path.join(datasetDir, folder, sub_folder))
    # print(sub_folder_files)#naama file txxdxx.jpg
    len_sub_folder = len(sub_folder_files) - 1
    for i, filename in enumerate(sub_folder_files):
      img = cv2.imread(os.path.join(datasetDir, folder, sub_folder, filename))
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
      resize = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
      imgs.append(resize)
      labels.append(normalize_label(folder,os.path.splitext(sub_folder)[0]))
      # descs.append(normalize_desc(folder, sub_folder))
      print_progress(i, len_sub_folder, folder, sub_folder, filename)

for img, label in zip(imgs, labels): 
  print(label)
  glcm_all_agls.append(
    calc_glcm_all_agls(img, label, props=properties))

for name in properties :
  for ang in angles:
    columns.append(name + "_" + ang)
columns.append("label")

# dataframe panda GLCM
glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)
#save to csv
glcm_df.to_csv("glcm_telur_dataset.csv")