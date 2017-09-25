import math
import numpy as np
import pickle
from os.path import join



def _shuffle_data(data):
  idx = np.arange(data[0].shape[0])
  np.random.shuffle(idx)
  shuffled_data = []
  for d in data:
    if type(d) == np.ndarray:
      # d = np.ascontiguousarray(d[idx])
      d = d[idx]
    else:
      d = [d[i] for i in idx]
    shuffled_data.append(d)
  return shuffled_data


class Dataset():
  class_info = [['road', [128,64,128]],
                ['building', [70,70,70]],
                ['infrastructure', [220,220,0]],
                ['nature', [107,142,35]],
                ['sky', [70,130,180]],
                ['person', [220,20,60]],
                ['vehicle', [0,0,142]]]
  num_classes = len(class_info)

  def __init__(self, split_name, batch_size, shuffle=True):
    # self.mean = np.array([116.5, 112.1, 104.1])
    # self.std = np.array([57.92, 56.97, 58.54])
    self.mean = np.array([75.205, 85.014, 75.089])
    self.std = np.array([46.894, 47.633, 46.471])
    self.batch_size = batch_size
    self.shuffle = shuffle
    # load the dataset
    # data_dir = '/home/kivan/datasets/SSDS/cityscapes'
    data_dir = 'local/data/'
    data = pickle.load(open(join(data_dir, split_name+'.pickle'), 'rb'))
    self.x = data['rgb']
    self.y = data['labels']
    self.names = data['names']
    # self.x = (self.x.astype(np.float32) - self.mean) / self.std

    self.num_examples = self.x.shape[0]
    self.height = self.x.shape[1]
    self.width = self.x.shape[2]
    self.channels = self.x.shape[3]
    self.num_batches = math.ceil(self.num_examples / self.batch_size)

  def __iter__(self):
    if self.shuffle:
      self.x, self.y, self.names = _shuffle_data([self.x, self.y, self.names])
    self.cnt = 0
    return self

  def __next__(self):
    if self.cnt >= self.num_batches:
      raise StopIteration
    offset = self.cnt * self.batch_size
    x = self.x[offset:offset+self.batch_size]
    y = self.y[offset:offset+self.batch_size]
    x = (np.ascontiguousarray(x).astype(np.float32) - self.mean) / self.std
    y = np.ascontiguousarray(y).astype(np.int32)
    names = self.names[offset:offset+self.batch_size]
    self.cnt += 1
    return x, y, names
  
  def get_img(self, name):
    i = self.names.index(name)
    img = self.x[i]
    img = img * self.std + self.mean
    img[img>255] = 255
    img[img<0] = 0
    return img.astype(np.uint8)