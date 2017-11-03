
from  __future__ import division, print_function
import cv2
import os
import sys
import util
import tensorflow as tf
import numpy as np

data_path=sys.argv[1]
save_data_path=sys.argv[2]
batch_size=1
IMG_W=int(sys.argv[3])
IMG_H=int(sys.argv[4])
ratio=float(sys.argv[5])

tra_image,tra_label=util.get_files(data_path,ratio)
tra_image_batch,tra_label_batch=util.get_batch(tra_image,tra_label,IMG_W,IMG_H,batch_size,method='train')

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        
       i=0
       while not coord.should_stop() and i<len(tra_image):
         image=sess.run([tra_image_batch])[0]
         image=np.reshape(image,[IMG_H,IMG_W,3])
         #print(image.shape)
         name='image_pro_%d'%i+'.jpg'
         cv2.imwrite(os.path.join(save_data_path,name),image)
         i+=1   
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)

#python Image_process.py /home/ye/user/yejg/database/eye_jpg/train/sick/ /home/ye/user/yejg/database/eye_jpg/train/gen_sick/ 1080 1080 1.0

