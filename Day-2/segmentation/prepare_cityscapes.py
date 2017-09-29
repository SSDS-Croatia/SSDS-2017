import sys
import os
import pickle
from os.path import join

import numpy as np
import PIL.Image as pimg
from tqdm import trange


def prepare_dataset(subset):
  classes = [['road', [0, 1]],
            #  ['building+infrastructure', [2,3,4,5,6,7]],
             ['building', [2,3]],
             ['infrastructure', [4,5,6,7]],             
             ['vegetation', [8,9]],
             ['sky', [10]],
             ['person', [11,12]],
             ['vehicle', [13,14,15,16,17,18]],
             ['ignore', [19]]]
  num_classes = len(classes) - 1
  ignore_id = num_classes
  id_map = {}
  for i, (_, ids) in enumerate(classes):
    for id in ids:
      id_map[id] = i

  id_map = {}
  for i, (_, ids) in enumerate(classes):
    for id in ids:
      id_map[id] = i

  save_dir = '/home/kivan/datasets/SSDS/cityscapes/final'

  img_size = (384, 160)
  cx_start = 0
  cx_end = 2048
  cy_start = 30
  cy_end = 900

  img_paths = []
  label_paths = []
  img_names = []
  root_dir = '/home/kivan/datasets/Cityscapes/2048x1024'
  img_dir = join(root_dir, 'rgb')
  labels_dir = join(root_dir, 'labels')
  mean = (75.205, 85.014, 75.089)
  std = (46.894, 47.633, 46.471)
  subset_dir = join(img_dir, subset)
  cities = next(os.walk(subset_dir))[1]
  for city in cities:
    files = next(os.walk(join(subset_dir, city)))[2]
    img_names.extend([f[:-4] for f in files])
    img_paths.extend([join(subset_dir, city, f) for f in files])
    label_paths.extend([join(labels_dir, subset, city, f[:-3]+'png') for f in files])

  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(join(save_dir, 'rgb'), exist_ok=True)
  os.makedirs(join(save_dir, 'labels'), exist_ok=True)
  mean_sum = np.zeros(3, dtype=np.float64)
  std_sum = np.zeros(3, dtype=np.float64)
  all_images = []
  all_labels = []
  for i in trange(len(img_names)):
    labels = pimg.open(label_paths[i])
    labels = labels.crop((cx_start,cy_start,cx_end,cy_end))
    labels = labels.resize(img_size, pimg.NEAREST)
    labels = np.array(labels, dtype=np.int32)
    ids = np.unique(labels)
    for id in ids:
      labels[labels==id] = id_map[id]
    labels = labels.astype(np.uint8)

    img = pimg.open(img_paths[i])
    img = img.crop((cx_start,cy_start,cx_end,cy_end))
    img = img.resize(img_size, pimg.LANCZOS)
    img = np.array(img, dtype=np.uint8)

    # compute mean
    mean_sum += img.mean((0,1))
    std_sum += img.std((0,1))

    # print(join(save_dir, 'rgb', img_name+'.png'))
    pimg.fromarray(img).save(join(save_dir, 'rgb', img_names[i]+'.png'))
    all_images.append(img)
    all_labels.append(labels)

  num_imgs = len(all_images)
  print('Num images: ', num_imgs)
  print('mean = ', mean_sum / num_imgs)
  print('std = ', std_sum / num_imgs)
  data = {}
  # all_images = np.stack(all_images)
  all_images = np.stack(all_images)
  all_labels = np.stack(all_labels)
  # print(all_images.shape)
  data['rgb'] = all_images
  data['labels'] = all_labels
  data['names'] = img_names
  pickle.dump(data, open(join(save_dir, subset+'.pickle'), 'wb'))
  # np.save(open(join(save_dir, subset+'2.npz'), 'wb'), data)
  # np.save(open(join(save_dir, subset+'4.npz'), 'wb'), all_images, allow_pickle=False)
  np.save(join(save_dir, subset+'_images.npz'), all_images)
  np.save(join(save_dir, subset+'_labels.npz'), all_labels)
  # all_labels.save(join(save_dir, subset+'_tagets.npz'))
  # np.save(open(join(save_dir, subset+'_names.npz'), 'wb'), data)
  # pickle.dump(data, open(join(save_dir, subset+'_names.pickle'), 'wb'))
  pickle.dump(img_names, open(join(save_dir, subset+'_names.pickle'), 'wb'))
  
if __name__ == '__main__':
  prepare_dataset('train')
  prepare_dataset('val')