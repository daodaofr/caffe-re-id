#!/usr/bin/env python
import sys
sys.path.append('../../../python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy
from scipy.io import loadmat

fea_path = 'fea/lbp_color/'
test_frames = 10
train_frames = 10
test_buffer = 1
train_buffer = 8

class sequenceGeneratorVideo(object):
  def __init__(self, buffer_size, clip_length, num_videos, video_dict, video_order):
    self.buffer_size = buffer_size
    self.clip_length = clip_length
    self.N = self.buffer_size*self.clip_length
    self.fea_dim = 58950
    self.num_videos = num_videos
    self.video_dict = video_dict
    self.video_order = video_order
    self.idx = 0

  def __call__(self):
    label_r = []
    color_fea_r=[]
    im_paths = []
 
    if self.idx + self.buffer_size >= self.num_videos:
      idx_list = range(self.idx, self.num_videos)
      idx_list.extend(range(0, self.buffer_size-(self.num_videos-self.idx)))
    else:
      idx_list = range(self.idx, self.idx+self.buffer_size)
    

    for i in idx_list:
      key = self.video_order[i]
      label = self.video_dict[key]['label']
    
      label_r.extend([label]*self.clip_length)

      tmp_color_fea = self.video_dict[key]['color_fea']
      tmp_fea = tmp_color_fea['tmp_fea']
 
      rand_frame = int(random.random()*(self.video_dict[key]['num_frames']-1-self.clip_length)+1)

      for i in range(rand_frame,rand_frame+self.clip_length):
        color_fea_r.append(tmp_fea[i])
    
    im_info = zip(im_paths)
  
    self.idx += self.buffer_size
    if self.idx >= self.num_videos:
      self.idx = self.idx - self.num_videos

    return label_r, im_info, color_fea_r
  

def advance_batch(result, sequence_generator, pool):
    label_r, im_info, color_fea_r = sequence_generator()
    result['label'] = label_r
    result['color_fea'] = np.asarray(color_fea_r)
    cm = np.ones(len(label_r))
    cm[0::10] = 0
    result['clip_markers'] = cm

class BatchAdvancer():
    def __init__(self, result, sequence_generator, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.pool)

class videoRead(caffe.Layer):

  def initialize(self):
    self.train_or_test = 'test'
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip 16
    self.N = self.buffer_size*self.frames
    self.fea_dim = 58950
    self.idx = 0
    self.video_list = 'train_lstm.txt' 
    
  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.video_list, 'r')
    f_lines = f.readlines()
    f.close()
    random.shuffle(f_lines)

    video_dict = {}
    current_line = 0
    self.video_order = []
    for ix, line in enumerate(f_lines):
      video = line.split(' ')[0].split('/')[1]
      image_name = line.split(' ')[0].split('/')[0]
      l = int(line.split(' ')[1])
      video_dict[video] = {}
      video_dict[video]['label'] = l
      video_dict[video]['color_fea'] = loadmat(fea_path+video+'.mat')
      mat1=loadmat(fea_path+video+'.mat')['tmp_fea']
      video_dict[video]['num_frames'] = mat1.shape[0]+1
      self.video_order.append(video) 

    self.video_dict = video_dict
    self.num_videos = len(video_dict.keys())
    shape = (self.N) 
    self.thread_result = {}
    self.thread = None
    pool_size = 24

    self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.frames, self.num_videos, self.video_dict, self.video_order)
    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.pool)
    self.dispatch_worker()
    self.top_names = ['label','clip_markers', 'color_fea']
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'label':
        shape = (self.N,)
      elif name == 'clip_markers':
        shape = (self.N,)
      elif name == 'color_fea':
        shape = (self.N, self.fea_dim)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
    new_result_label = [None]*len(self.thread_result['label']) 
    new_result_cm = [None]*len(self.thread_result['clip_markers'])
    new_result_cf = [None]*len(self.thread_result['color_fea'])
    for i in range(self.frames):
      for ii in range(self.buffer_size):
        old_idx = ii*self.frames + i
        new_idx = i*self.buffer_size + ii
        new_result_label[new_idx] = self.thread_result['label'][old_idx]
        new_result_cm[new_idx] = self.thread_result['clip_markers'][old_idx]
        new_result_cf[new_idx] = self.thread_result['color_fea'][old_idx]

    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'label':
        top[top_index].data[...] = new_result_label
      elif name == 'clip_markers':
        top[top_index].data[...] = new_result_cm
      elif name == 'color_fea':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_cf[i] 

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

class videoReadTrain_RGB(videoRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.fea_dim=58950
    self.idx = 0
    self.video_list = 'train_lstm.txt' 

class videoReadTest_RGB(videoRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.fea_dim=58950
    self.idx = 0
    self.video_list = 'train_lstm.txt' 


