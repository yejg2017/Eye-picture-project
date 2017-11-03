from  __future__ import division, print_function
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import os
import math
import cv2
import matplotlib.pyplot as plt



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


class resize:
   def __init__(self,
                res_img_h,
                res_img_w):
      #self.image_path=image_path
      self.res_img_h=res_img_h
      self.res_img_w=res_img_w
      
   def crop_resize(self,image_path):
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    if width == height:
        resized_image = cv2.resize(image, (self.res_img_h,self.res_img_w))
    elif width > height:
        resized_image = cv2.resize(image, (int(width * float(self.res_img_h)/height), self.res_img_w))
        cropping_length = int( (resized_image.shape[1] - self.res_img_h) / 2)
        resized_image = resized_image[:,cropping_length:cropping_length+self.res_img_w]
    else:
        resized_image = cv2.resize(image, (self.res_img_h, int(height * float(self.res_img_w)/width)))
        cropping_length = int( (resized_image.shape[0] - self.res_img_w )/ 2)
        resized_image = resized_image[cropping_length:cropping_length+self.res_img_h, :]

    return (resized_image - 127.5) / 127.5



def get_batch(image, label, image_W, image_H, batch_size, capacity):
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
    image = tf.image.decode_jpeg(image_contents, channels=3,
                                 try_recover_truncated=True,
                                 acceptable_fraction=0.5)



    # data argumentation here?
    
    image = tf.image.resize_images(image, [image_W, image_H])    
        
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                batch_size= batch_size,
                                num_threads= 64, 
                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


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



