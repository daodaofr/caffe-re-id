#from classify_video import *
import numpy as np
import glob
caffe_root = '../../../'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
import pickle
import h5py
import random
from scipy.io import loadmat

color_path = 'lbp_color'
fea_dim = 58950
lstm_model = 'deploy_lstm.prototxt'
RGB_lstm = 're-id.caffemodel'
RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

def LRCN_ex_fea(net, mat_path):
  clip_length = 10
  offset = 1
  output_predictions = np.zeros((clip_length,512))
  color_fea_input = loadmat(mat_path)
  mat1=color_fea_input['tmp_fea']
  num_frames = mat1.shape[0]
  color_fea_r = color_fea_input['tmp_fea']
  caffe_in = np.zeros((clip_length, fea_dim))
  clip_clip_markers = np.ones((clip_length,1,1,1))
  clip_clip_markers[0:1,:,:,:] = 0
  f = random.randint(0,1)
  rand_frame = int(random.random()*(num_frames-clip_length)+1)
  for i in range(1):
    k=0
    for j in range(rand_frame, rand_frame+clip_length):
      caffe_in[k] = color_fea_r[j]
      k=k+1
    out = net.forward_all(color_fea=caffe_in.reshape((clip_length, fea_dim, 1, 1)), clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['lstm1'],1)
  return output_predictions

video_list = 'train_lstm.txt'
f = open(video_list, 'r')
f_lines = f.readlines()
f.close()
true_pred = 0
all_test = 0
all_fea = np.zeros((len(f_lines), 10,512))
itr = 1
for it in range(itr):
    for ix, line in enumerate(f_lines):
        video = line.split(' ')[0]
        l = int(line.split(' ')[1])
        video1 = line.split(' ')[0].split('/')[1]
        color_mat_path = color_path+video1+'.mat'
        print "processing the %d th image" % ix
        tmp_fea = \
             LRCN_ex_fea( RGB_lstm_net, color_mat_path)
        all_fea[ix] = tmp_fea
    f_all = h5py.File('train_25k_'+str(it)+'.h5', "w")
    f_all.create_dataset('train_set', data = all_fea)
    f_all.close()

