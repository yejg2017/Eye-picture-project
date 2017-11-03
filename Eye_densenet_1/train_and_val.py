import tensorflow as tf
import model
import random
import os
import util
import optparse
import numpy as np
import cv2
import tensorlayer as tl
#%%
paser=optparse.OptionParser(usage='Specify paramenters for DesNet model')

paser.add_option('-B',dest='BN_EPSILON',type='float',default=0.001)
#
paser.add_option('-w',dest='weight_decay',type='float',default=0.0005)
#
#paser.add_option('--fw',dest='fc_weight_decay',type='float',default=1e-3)

paser.add_option('--mo',dest='nesterov_momentum',type='float',default=1e-4)

paser.add_option('-n',dest='n_classes',type='int',default=2)

paser.add_option('-b',dest='batch_size',type='int',default=32)

paser.add_option('-p',dest='per_block_num',type='int',default=3)

paser.add_option('--gw',dest='IMG_W',type='int',default=32)

paser.add_option('--gh',dest='IMG_H',type='int',default=32)

#paser.add_option('-c',dest='capacity',type='int',default=256)

paser.add_option('--lr',dest='learning_rate',type='float',default=0.0002)

#paser.add_option('-f',dest='train_dir',type='str')

paser.add_option('-o',dest='output',type='str')
paser.add_option('-s',dest='model_path',type='str')

paser.add_option('-u',dest='RATIO',type='float',default=0.2,help='specify number for ratio of evalute')

paser.add_option('-r',dest='reuse',action="store_false",default=False)
paser.add_option('-d',dest='data_path',type='str')
paser.add_option('-M',dest='MAX_STEP',type='int',default=10000)
#paser.add_option('-k',dest='pkeep',type='float',default=0.5)

#%%
##   paramenters
(options,arg)=paser.parse_args()
BN_EPSILON=options.BN_EPSILON
weight_decay=options.weight_decay
#fc_weight_decay=options.fc_weight_decay
model.BN_EPSILON=options.BN_EPSILON
model.weight_decay=options.weight_decay

num_labels=options.n_classes
batch_size=options.batch_size
per_block_num=options.per_block_num

IMG_W=options.IMG_W
IMG_H=options.IMG_H
learning_rate=options.learning_rate
reuse=options.reuse
nesterov_momentum=options.nesterov_momentum
MAX_STEP=options.MAX_STEP

outfile=options.output
model_path=options.model_path
ratio=options.RATIO
data_path=options.data_path
#pkeep=options.pkeep

train,train_labels,val,val_labels=util.read_files(data_path,ratio)

train_image_batch,train_label_batch=util.get_batch(train,train_labels,IMG_W,IMG_H,batch_size,method='test')
val_image_batch,val_label_batch=util.get_batch(val,val_labels,IMG_W,IMG_H,batch_size,method='test')

x=tf.placeholder(dtype=tf.float32,shape=[batch_size,IMG_H,IMG_W,3])
y_=tf.placeholder(dtype=tf.int32,shape=[batch_size])
keep_prob=tf.placeholder(dtype=tf.float32)



logits=model.inference(x,per_block_num,num_labels,reuse,keep_prob)
predictions=tf.nn.softmax(logits)
loss=model.loss(logits,y_)
acc=model.accuracy(predictions,y_)
#train_op=model.train_op(loss,learning_rate)
#l2_loss=tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

optimizer=tf.train.MomentumOptimizer(learning_rate,nesterov_momentum,use_nesterov=True)

train_op=optimizer.minimize(loss+l2_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    saver = tf.train.Saver()
    outfile_split=outfile.split('/')
    if outfile_split[-1]=='':
       outfile=outfile
    else:
       outfile=outfile+'/'


    model_path_split=model_path.split('/')
    if model_path_split[-1]=='':
       model_path=model_path
    else:
       model_path=model_path+'/'

    p=open(outfile+'BN%d_mo%f_lr%f_bs%d_wd%f_gW%d_gH%d'%(BN_EPSILON,nesterov_momentum,learning_rate,batch_size,weight_decay,IMG_W,IMG_H)+'.txt','w')
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
    
      for step in range(MAX_STEP):
        if coord.should_stop():
           break
        tra_image,tra_label=sess.run([train_image_batch,train_label_batch])
        tra_acc,tra_loss,_=sess.run([acc,loss,train_op],
                                       feed_dict={x:tra_image,y_:tra_label,keep_prob:0.5})
       
        if step%50==0:
           
           print('Step %d, train loss = %.2f, train accuracy = %.2f%%\n' %(step, tra_loss, tra_acc*100.0))
           p.write('Step %d, train loss = %.2f, train accuracy = %.2f%%\n' %(step, tra_loss, tra_acc*100.0))
     
        if (step%100==0) or (step+1==MAX_STEP):
           val_image,val_label=sess.run([val_image_batch,val_label_batch])
           val_acc,val_loss=sess.run([acc,loss],
                                      feed_dict={x:val_image,y_:val_label,keep_prob:1.0})
          
           print('*** Step %d, val loss = %.2f, val accuracy = %.2f%% ***\n' %(step, val_loss, val_acc*100.0))
           p.write('*** Step %d, val loss = %.2f, val accuracy = %.2f%% ***\n' %(step, val_loss, val_acc*100.0))

        if step % 2000 == 0 or (step + 1) == MAX_STEP:
           style='BN%d_mo%f_lr%f_bs%d_wd%f_gW%d_gH%d/'%(BN_EPSILON,nesterov_momentum,learning_rate,batch_size,weight_decay,IMG_W,IMG_H)
           logs_train_dir=model_path+style
           checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
           saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
          print('Done training -- epoch limit reached')
    finally:
          coord.request_stop()
    coord.join(threads)
    p.close()


