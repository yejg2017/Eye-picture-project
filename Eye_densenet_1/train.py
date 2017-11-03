import tensorflow as tf
import numpy as np
import dense_net as dnet
import util
import optparse
import tensorlayer as tl
import os



parse=optparse.OptionParser(usage='Paramenter for model')

parse.add_option('-b',dest='batch_size',type='int',default=32,help='specify number')

parse.add_option('--lr',dest='learning_rate',type='float',default=0.0001,help='specify number')

parse.add_option('-f',dest='model_path',type='str',default='/home/ye/user/yejg/project/Eye_densenet_1/model_path/',
                help='path of saving model')

parse.add_option('-M',dest='MAX_STEP',type='int',default=20000,help='specify number for epochs')

parse.add_option('-H',dest='IMG_H',type='int',default=208,help='specify number for image height')

parse.add_option('-W',dest='IMG_W',type='int',default=208,help='specify number for image width')

parse.add_option('-r',dest='RATIO',type='float',default=0.2,help='specify number for ratio of evalute')

parse.add_option('-o',dest='outfile',type='str',default='/home/ye/user/yejg/project/Eye_densenet_1/result/',help='file for result')

parse.add_option('-g',dest='growth_rate',type='int',default=12,help='specify number for growth_rate')

parse.add_option('-t',dest='total_blocks',type='int',default=3,help='specify number for total blocks')

parse.add_option('-d',dest='depth',type='int',default=40,help='specify number for depth')

parse.add_option('--rd',dest='reduction',type='float',default=0.5)

parse.add_option('-k',dest='keep_prob',type='float',default=0.5)

parse.add_option('-w',dest='weight_decay',type='float',default=1e-4)

parse.add_option('-n',dest='nesterov_momentum',type='float',default=0.9)
parse.add_option('-m',dest='model_type',type='str',default='dense_net')
parse.add_option('-D',dest='data_path',type='str',default='/home/ye/user/yejg/database/eye_jpg/train/')
(options,args)=parse.parse_args()


batch_size=options.batch_size
learning_rate=options.learning_rate
n_classes=2
growth_rate=options.growth_rate
total_blocks=options.total_blocks
depth=options.depth
reduction=options.reduction
keep_prob=options.keep_prob
weight_decay=options.weight_decay
nesterov_momentum=options.nesterov_momentum
img_w=options.IMG_W
img_h=options.IMG_H
model_type=options.model_type
MAX_STEP=options.MAX_STEP
outfile=options.outfile  #  store reuslt(acc,loss)
data_path=options.data_path
model_path=options.model_path
ratio=options.RATIO


x=tf.placeholder(tf.float32,shape=[batch_size,img_h,img_w,3])
y_=tf.placeholder(tf.int32,shape=[batch_size])
kprob=tf.placeholder(tf.float32)



train,train_labels,val,val_labels=util.read_files(data_path,ratio)

train_image_batch,train_label_batch=util.get_batch(train,train_labels,img_w,img_h,batch_size,method='test')
val_image_batch,val_label_batch=util.get_batch(val,val_labels,img_w,img_h,batch_size,method='test')



dense_net=dnet.DenseNet(images=x,labels=y_,n_classes=n_classes,
                        growth_rate=growth_rate,total_blocks=total_blocks,
                        depth=depth,reduction=reduction,
                        keep_prob=kprob,weight_decay=weight_decay,
                        nesterov_momentum=nesterov_momentum,model_type=model_type,
                        learning_rate=learning_rate,is_training=True)

dense_net._build_graph()

loss=dense_net.cross_entropy
acc=dense_net.accuracy
train_op=dense_net.train_step


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

    p=open(outfile+'lr%f_bs%d_dp%d_tb%d_wd%f_nm%f_gW%d_gH%d'%(learning_rate,batch_size,depth,total_blocks,weight_decay,nesterov_momentum,img_w,img_h)+'.txt','w')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:

      for step in range(MAX_STEP):
        if coord.should_stop():
           break
        tra_image,tra_label=sess.run([train_image_batch,train_label_batch])
        tra_acc,tra_loss,_=sess.run([acc,loss,train_op],
                                       feed_dict={x:tra_image,y_:tra_label,kprob:0.5})

        if step%50==0:

           print('Step %d, train loss = %.2f, train accuracy = %.2f%%\n' %(step, tra_loss, tra_acc*100.0))
           p.write('Step %d, train loss = %.2f, train accuracy = %.2f%%\n' %(step, tra_loss, tra_acc*100.0))

        if (step%200==0) or (step+1==MAX_STEP):
           val_image,val_label=sess.run([val_image_batch,val_label_batch])
           val_acc,val_loss=sess.run([acc,loss],
                                      feed_dict={x:val_image,y_:val_label,kprob:1.0})

           print('*** Step %d, val loss = %.2f, val accuracy = %.2f%% ***\n' %(step, val_loss, val_acc*100.0))
           p.write('*** Step %d, val loss = %.2f, val accuracy = %.2f%% ***\n' %(step, val_loss, val_acc*100.0))

        if step % 2000 == 0 or (step + 1) == MAX_STEP:

           style='lr_%f_bc_%d_imw_%d_imh_%d/'%(learning_rate,batch_size,img_w,img_h)
           logs_train_dir=model_path+style
           checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
           saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
          print('Done training -- epoch limit reached')
    finally:
          coord.request_stop()
    coord.join(threads)
    p.close()




