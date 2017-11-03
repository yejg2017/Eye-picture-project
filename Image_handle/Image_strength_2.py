import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import optparse


parse=optparse.OptionParser(usage='Paramenter for Images')
parse.add_option('-d',dest='data_path',type='str',default='/home/ye/user/yejg/database/eye_tang/pictures/train/sick/')
parse.add_option('-s',dest='save_path',type='str',default='/home/ye/user/yejg/database/eye_tang/pictures/test/')
parse.add_option('-W',dest='cropped_width',type='int',default=256)
parse.add_option('-H',dest='cropped_height',type='int',default=256)

(options,args)=parse.parse_args()
data_path=options.data_path
save_image_path=options.save_path
cropped_height=options.cropped_height
cropped_width=options.cropped_width



class Image_Strength:
    def __init__(self,imgh_cropped,imgw_cropped,training=True):
        self.imgh_cropped=imgh_cropped
        self.imgw_cropped=imgw_cropped
        self.training=training
        self.num_channels=3
     
    def read_image(self,path):
        image=cv2.imread(path)
        return image
    
    def read_path(self,path):
        self.image_list=[os.path.join(path,f) for f in os.listdir(path)]
     
    def random_show(self):
        random_image=np.random.choice(self.image_list)
        image=cv2.imread(random_image)
        #self.height,self.width,self.num_channels=image.shape
        plt.imshow(image)
        plt.show()
    
    def pre_process_image(self,image):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

      if self.training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        # image = tf.random_crop(image, size=[self.imgh_cropped, self.imgw_cropped, self.num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        image = tf.random_crop(image, size=[self.imgh_cropped, self.imgw_cropped, self.num_channels])
        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
        #image=tf.image.rgb_to_grayscale(image)
      else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=self.imgh_cropped,
                                                       target_width=self.imgw_cropped)
        #image=tf.image.rgb_to_grayscale(image)
      return image


Image_S=Image_Strength(cropped_height,cropped_width,training=True)
Image_S.read_path(data_path)


with tf.Session() as sess:
     for f in Image_S.image_list:
            image=Image_S.read_image(f)
            x=tf.placeholder(tf.float32,shape=[image.shape[0],image.shape[1],image.shape[2]])
            crop_image=Image_S.pre_process_image(x)
    
            cropped_image=sess.run([crop_image],feed_dict={x:image})[0]
            cv2.imwrite(os.path.join(save_image_path,os.path.basename(f)),cropped_image)



     
