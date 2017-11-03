# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import cv2
 #%%
train_dir='/home/ye/user/yejg/database/eye_jpg/train/'

def read_files(path,ratio=0.2):
    health=[]
    sick=[]
    health_label=[]
    sick_label=[]
    for d in os.listdir(path):
        for f in os.listdir(os.path.join(path,d)):
            if d=='health':
                health.append(os.path.join(path,'health/',f))
                health_label.append(1)
            if d=='sick':
                sick.append(os.path.join(path,'sick/',f))
                sick_label.append(0)
#            if d=='generate_sick':
#                sick.append(os.path.join(path,'generate_sick/',f))
#                sick_label.append(0)
    #return health_files,health,sick_files,sick
    print('There are %d health\nThere are %d sick' %(len(health), len(sick)))

    image_list = np.hstack((health, sick))
    label_list = np.hstack((health_label, sick_label))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio)) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples

    n_train = int(n_train)
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images,tra_labels,val_images,val_labels

def get_files(path,ratio=0.5):
    images_path_list=[]
    path_split=path.split('/')
    if path_split[-1]!='':
       path=path+'/'
    if path_split[-1]=='':
       path=path
    for f in os.listdir(path):
        images_path_list.append(os.path.join(path,f))
    
    print('There are %d images:'%(len(images_path_list)))
    temp=np.array(images_path_list)
    np.random.shuffle(temp)
    
    n_samples=int(math.ceil(len(images_path_list)*ratio))
    images=temp[:n_samples]
    labels=[1]*len(images)
    
    return images,labels


#images,labels=get_files('/home/ye/user/yejg/database/eye_jpg/train/health')
#print(images[:10])

def get_batch(image, label, image_W, image_H, batch_size, capacity=256,method='train'):

    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    if method=='test':

#       image=tf.image.random_brightness(image,max_delta=63) 
#       image=tf.image.random_contrast(image,lower=0.2,upper=1.8) 
       image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)

    if method=='train':
#      image=tf.image.resize_images(image,[image_W,image_H])
       image=tf.image.adjust_brightness(image,0.2)
       image=tf.image.adjust_saturation(image, 0.4) 
#       image=tf.image.adjust_contrast(image,-0.5) 
#       image=tf.image.adjust_hue(image, 0.7)
       image=tf.image.flip_up_down(image)
       image=tf.image.random_flip_left_right(image)
       image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)    
    # if you want to test the generated batches of images, you might want to comment the following line.
    
#       image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

BATCH_SIZE=1
IMG_H=208
IMG_W=208
CAPACITY=256
#
#
tra_images, tra_labels,_,_= read_files(train_dir)
#val_images,val_labels=get_files('/home/ye/user/yejg/database/eye_jpg/train/health')
tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#val_image_batch,val_label_batch=get_batch(val_images,val_labels,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
#
with tf.Session() as sess:
    i=0
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            img,label=sess.run([tra_image_batch,tra_label_batch])
#            val,v_label=sess.run([val_image_batch,val_label_batch])
            print(img.shape,label)
#            print(val.shape,v_label)            
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
            i+=1
#           
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
#    
#
#def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle=True):
#    # 创建一个混排样本的队列，然后从样本队列中读取 'batch_size'数量的 images + labels数据（每个样本都是由images + labels组成）
#    num_preprocess_threads=16 # 预处理采用多线程
#    if shuffle:
#        images,label_batch=tf.train.shuffle_batch(
#            [image,label],
#            batch_size=batch_size,
#            num_threads=num_preprocess_threads,
#            capacity=min_queue_examples+3*batch_size
#        )
#    else:
#        images,label_batch,=tf.train.batch(
#            [image,label],
#            batch_size=batch_size,
#            num_threads=num_preprocess_threads,
#            capacity=min_queue_examples
#        )
#    tf.summary.image('images',images) #训练图像可视化
#    return  images,tf.reshape(label_batch,[batch_size])            
#
#
#
#
def check_transf(images_list):
    png=[]
    jpeg=[]
    for img in images_list:
        if img.endswith('.png'):
            png.append(img)
        if img.endswith('.jpg'):
            jpeg.append(img)
    return png,jpeg




def plot_images(images):
    images = images[0:9]
    fig, axes = plt.subplots(3, 3)
    axes = np.ravel(axes)
    for i in range(len(images)):
        if len(images[i].shape) == 2:
            axes[i].imshow(images[i], cmap="gray")
        else:
            axes[i].imshow(images[i], interpolation="nearest")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def get_next_image_loc(imgdir):
    for root, dirs, files in os.walk(imgdir):
        for name in files:
            path = os.path.join(root, name).split(os.path.sep)[::-1]
            yield (path[1], path[0])


def compute_edges(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobel_y = np.uint8(np.absolute(sobel_y))
    edged = cv2.bitwise_or(sobel_x, sobel_y)
    return edged


def crop_image_to_edge(image, threshold=10, margin=0.2):
    edged = compute_edges(image)
    # find edge along center and crop
    mid_y = edged.shape[0] // 2
    notblack_x = np.where(edged[mid_y, :] >= threshold)[0]
    if notblack_x.shape[0] == 0:
        lb_x = 0
        ub_x = edged.shape[1]
    else:
        lb_x = notblack_x[0]
        ub_x = notblack_x[-1]
    if lb_x > margin * edged.shape[1]:
        lb_x = 0
    if (edged.shape[1] - ub_x) > margin * edged.shape[1]:
        ub_x = edged.shape[1]
    mid_x = edged.shape[1] // 2
    notblack_y = np.where(edged[:, mid_x] >= threshold)[0]
    if notblack_y.shape[0] == 0:
        lb_y = 0
        ub_y = edged.shape[0]
    else:
        lb_y = notblack_y[0]
        ub_y = notblack_y[-1]
    if lb_y > margin * edged.shape[0]:
        lb_y = 0
    if (edged.shape[0] - ub_y) > margin * edged.shape[0]:
        ub_y = edged.shape[0]
    cropped = image[lb_y:ub_y, lb_x:ub_x, :]
    return cropped


def crop_image_to_aspect(image, tar=1.2):
    # load image
    image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # compute aspect ratio
    h, w = image_bw.shape[0], image_bw.shape[1]
    sar = h / w if h > w else w / h
    if sar < tar:
        return image
    else:
        k = 0.5 * (1.0 - (tar / sar))
        if h > w:
            lb = int(k * h)
            ub = h - lb
            cropped = image[lb:ub, :, :]
        else:
            lb = int(k * w)
            ub = w - lb
            cropped = image[:, lb:ub, :]
        return cropped


def brighten_image_hsv(image, global_mean_v):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image_hsv)
    mean_v = int(np.mean(v))
    v = v - mean_v + global_mean_v
    image_hsv = cv2.merge((h, s, v))
    image_bright = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image_bright


def brighten_image_rgb(image, global_mean_rgb):
    r, g, b = cv2.split(image)
    m = np.array([np.mean(r), np.mean(g), np.mean(b)])
    brightened = image + global_mean_v - m
    return brightened



def image_pre_train(path,method='hsv'):

    if method=='hsv':
       vs=[]
       for f in path:
         image=cv2.imread(f)

         image_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
         h,s,v=cv2.split(image_hsv)
         vs.append(np.mean(v))

       return int(np.mean(np.array(vs)))

    if method=='rgb':
       mean_rgbs=[]
       for f in path:
           image=cv2.imread(f)
           image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2BGR)
           r,g,b=cv2.split(image_rgb)
           mean_rgbs.append(np.array([np.mean(r),np.mean(g),np.mean(b)]))
       return np.mean(mean_rgbs,axis=0)

