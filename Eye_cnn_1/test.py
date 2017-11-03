#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:08:07 2017

@author: gr
"""
# test on R2,R3a,R3s
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import model

image_W = 208
image_H = 208
CAPACITY = 1000
BATCH_SIZE = 1
N_CLASSES = 2

test_dir = '/home/yanqinhong/DL/eye/data/test' # test data file
logs_train_dir = '/home/yanqinhong/DL/eye/logs/train/' # model file

test_image = []
test_label = []
for file1 in os.listdir(test_dir):
    for file2 in os.listdir(test_dir + '/' + file1):
        if file1 == 'health':
            test_label.append(0)
        else:
            test_label.append(1)
        test_image.append(test_dir+'/'+file1+'/'+file2)
    

print('There are %d test image' %len(test_image))

pred = []

with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, shape=[1, 208, 208, 3])
        logit = model.inference(x, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            for image in test_image:
                image = Image.open(image)
                image = image.resize((image_W, image_H))
                image = np.array(image)
                image = tf.convert_to_tensor(image)
                image = tf.image.per_image_standardization(image)
                image = tf.cast(image, tf.float32)
                image = tf.reshape(image, [1, 208, 208, 3])
                image = image.eval(session = sess)
                prediction = sess.run(logit, feed_dict={x:image})
                max_index = np.argmax(prediction)
                pred.append(max_index)

# testing accuracy
acc = np.sum(test_label==np.array(pred))/float(len(test_label))
print('testing accuracy: %s' %acc)
        
        