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
parse.add_option('-c',dest='color_ordering',type='int',default=0)

(options,args)=parse.parse_args()
data_path=options.data_path
save_image_path=options.save_path
cropped_height=options.cropped_height
cropped_width=options.cropped_width
color_ordering=options.color_ordering

#import tensorflow as tf
#import numpy as np
# import matplotlib.pyplot as plt

def distort_color(image, color_ordering=color_ordering):
    '''
    随机调整图片的色彩，定义两种处理顺序。
    注意，对3通道图像正常，4通道图像会出错，自行先reshape之
    :param image: 
    :param color_ordering: 
    :return: 
    '''
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    '''
    对图片进行预处理，将图片转化成神经网络的输入层数据。
    :param image: 
    :param height: 
    :param width: 
    :param bbox: 
    :return: 
    '''
    # 查看是否存在标注框。
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
   
    # 随机的截取图片中一个块。
#    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
#        tf.shape(image), bounding_boxes=bbox)
#    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
#        tf.shape(image), bounding_boxes=bbox)
#    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    #distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image

def pre_main(path,bbox=None):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image=cv2.imread(path)
    x=tf.placeholder(tf.float32,shape=[image.shape[0],image.shape[1],image.shape[2]])
    pre_image=preprocess_for_train(x,cropped_height,cropped_width,bbox)
    with tf.Session() as sess:
        cropped_image=sess.run([pre_image],feed_dict={x:image})[0]
        #print(cropped_image.shape)
        #plt.imshow(cropped_image)
        #plt.show()
        cv2.imwrite(os.path.join(save_image_path,os.path.basename(path)),cropped_image)
        #return cropped_image


image_list=[os.path.join(data_path,f) for f in os.listdir(data_path)]
for f in image_list:
    pre_main(f)

