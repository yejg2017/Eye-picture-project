import tensorflow as tf
import numpy as np
import os
import math

train_dir = '/home/yanqinhong/DL/eye/data/train/'

def get_files(file_dir, ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    health = []
    health_label = []
    sick = []
    sick_label = []
    for file1 in os.listdir(file_dir):
        for file2 in os.listdir(file_dir+'/'+file1):
            if file1 == 'health':
                health.append(file_dir+'health/'+file2)
                health_label.append(0)
            else:
                sick.append(file_dir+'sick/'+file2)
                sick_label.append(1)
    print('There are %d health\nThere are %d sick' %(len(health), len(sick)))
    
    image_list = np.hstack((health, sick))
    label_list = np.hstack((health_label, sick_label))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    
    n_train = int(n_train)
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
    return tra_images,tra_labels,val_images,val_labels


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

if __name__ is '__main__':
    import matplotlib.pyplot as plt
    
    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208
    ratio = 0.2
    
    tra_images, tra_labels, val_images, val_labels = get_files(train_dir, ratio)
    tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        try:
            while not coord.should_stop() and i<1:
                
                img, label = sess.run([tra_image_batch, tra_label_batch])
                
                # just test one batch
                for j in np.arange(BATCH_SIZE):
                    print('label: %d' %label[j])
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1
                
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
    





    
