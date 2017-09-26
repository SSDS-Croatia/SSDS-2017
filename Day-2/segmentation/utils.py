import time
from os.path import join

import tensorflow as tf
import numpy as np
import PIL.Image as pimg


def print_stats(conf_mat, name, class_info):
  num_correct = conf_mat.trace()
  num_classes = conf_mat.shape[0]
  total_size = conf_mat.sum()
  avg_pixel_acc = num_correct / total_size * 100.0
  TPFP = conf_mat.sum(0)
  TPFN = conf_mat.sum(1)
  FN = TPFN - conf_mat.diagonal()
  FP = TPFP - conf_mat.diagonal()
  class_iou = np.zeros(num_classes)
  print('\n'+name+'evaluation stats:')
  for i in range(num_classes):
    TP = conf_mat[i,i]
    class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
    class_name = class_info[i][0]
    print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
  avg_class_iou = class_iou.mean()
  print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
  print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
  return avg_class_iou


def colorize_labels(y, class_colors, save_path=None):
  width = y.shape[1]
  height = y.shape[0]
  y_rgb = np.zeros((height, width, 3), dtype=np.uint8)
  for cid in range(len(class_colors)):
    cpos = np.repeat((y == cid).reshape((height, width, 1)), 3, axis=2)
    cnum = cpos.sum() // 3
    y_rgb[cpos] = np.array(class_colors[cid][1] * cnum, dtype=np.uint8)
  if save_path:
    image = pimg.fromarray(y_rgb)
    image.save(save_path)
  return y_rgb


def draw_labels(y, names, class_info, save_dir):
  tf.gfile.MakeDirs(save_dir)
  for i, name in enumerate(names):
    img = colorize_labels(y[i], class_info)
    save_path = join(save_dir, name + '.png')
    pimg.fromarray(img).save(save_path)


def get_expired_time(start_time):
  curr_time = time.time()
  delta = curr_time - start_time
  hour = int(delta / 3600)
  delta -= hour * 3600
  minute = int(delta / 60)
  delta -= minute * 60
  seconds = delta
  return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds

def clear_dir(path):
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)