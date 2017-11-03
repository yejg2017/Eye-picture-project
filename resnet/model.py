# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import numpy as np
#%%
BN_EPSILON = 0.001


#   build ResNet model
#%%
def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name=x.op.name
    tf.summary.scalar(tensor_name+'/activation',x)
    tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x))
    
    
#%%

#  create weights variables
def create_variables(name,shape,initializer=tf.contrib.layers.xavier_initializer(),
                     is_fc_layers=False):
   
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    if is_fc_layers:
       regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)
    else:
       regularizer=None
       
    new_variable=tf.get_variable(name,shape,dtype=tf.float32,
                                 initializer=initializer,
                                 regularizer=regularizer)
    
    return new_variable
 
#%%

def batch_normalization_layer(input_layers,dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean,std=tf.nn.moments(input_layers,axes=[0,1,2])
    beta=tf.get_variable('beta',shape=dimension,dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0,tf.float32))
    gamma=tf.get_variable('gemma',shape=dimension,dtype=tf.float32,
                          initializer=tf.constant_initializer(1.0,tf.float32))
    
    bn_layers=tf.nn.batch_normalization(input_layers,mean=mean,variance=std,
                             offset=beta,scale=gamma,variance_epsilon=BN_EPSILON)
    
    return bn_layers
 
   
#%%

def output_layers(input_layers,num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    in_channel=input_layers.get_shape().as_list()[-1]
    
    fc_w=create_variables('weight',shape=[in_channel,num_labels],
                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0),is_fc_layers=True)
    
    fc_b=create_variables('biase',shape=[num_labels],
                          initializer=tf.zeros_initializer(tf.float32))
    
    out=tf.matmul(input_layers,fc_w)+fc_b
    return out    
 
   
#%%

def conv_bn_relu_layer(input_layer,filter_shape,stride):
   
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channle=filter_shape[-1]
    filter_=create_variables(name='conv',shape=filter_shape)
    conv_layer=tf.nn.conv2d(input_layer,filter=filter_,strides=[1,stride,stride,1],padding='SAME')
    
    bn_layer=batch_normalization_layer(conv_layer,out_channle)
    output=tf.nn.relu(bn_layer)
    return output
 
   
#%%
def bn_relu_conv_layer(input_layer,filter_shape,stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    
    in_channel=input_layer.get_shape().as_list()[-1]
    
    bn_layer=batch_normalization_layer(input_layer,in_channel)
    relu_layer=tf.nn.relu(bn_layer)
    
    filter_=create_variables(name='conv',shape=filter_shape)
    conv_layer=tf.nn.conv2d(relu_layer,filter=filter_,strides=[1,stride,stride,1],padding='SAME')
    return conv_layer

#%%

def residual_block(input_layer,output_channel,first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    
    input_channel=input_layer.get_shape().as_list()[-1]
    
    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel*2==output_channel:
       increase_dim=True
       stride=2
    elif input_channel==output_channel:
       increase_dim=False
       stride=1
       
    else:
       raise ValueError('Output and input channel do not match in residual block')
    
   # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
       if first_block:
          filter_=create_variables(name='conv',shape=[3,3,input_channel,output_channel])
          conv1=tf.nn.conv2d(input_layer,filter_,strides=[1,stride,stride,1],
                             padding='SAME')
          
       else:
          conv1=bn_relu_conv_layer(input_layer,filter_shape=[3,3,input_channel,output_channel],stride=stride)
          
    
    with tf.variable_scope('conv2_in_block'):
       conv2=bn_relu_conv_layer(conv1,filter_shape=[3,3,output_channel,output_channel],stride=1)
       
    
    if increase_dim is True:
       pooled_input=tf.nn.avg_pool(input_layer,ksize=[1,2,2,1],
                                   strides=[1,2,2,1],padding='VALID')
       
       pooled_input=tf.nn.lrn(pooled_input,depth_radius=4,bias=1.0,alpha=1e-4,beta=0.75)
       padded_input=tf.pad(pooled_input,paddings=[[0,0],[0,0],[0,0],[input_channel//2,input_channel//2]])
    
    else:
       padded_input=input_layer
      
    output=conv2+padded_input   #It should be identity map.I think
    return output   
    
 
#%%

#  build graph
def inference(input_tensor_batch,n,num_labels,reuse,pkeep):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
       
    layers=[]
    with tf.variable_scope('conv0',reuse=reuse):
         conv0=conv_bn_relu_layer(input_tensor_batch,[3,3,3,64],1)
         activation_summary(conv0)
         layers.append(conv0)
         
    for i in range(n):
       with tf.variable_scope('conv1_%d'%i,reuse=reuse):
          if i==0:
             conv1=residual_block(layers[-1],output_channel=128,first_block=True)
             
          else:
             conv1=residual_block(layers[-1],output_channel=128)
          activation_summary(conv1)
          layers.append(conv1)
          
    for i in range(n):
       with tf.variable_scope('conv2_%d'%i,reuse=reuse):
          conv2=residual_block(layers[-1],output_channel=256)
          activation_summary(conv2)
          layers.append(conv2)
          
    for i in range(n):
       with tf.variable_scope('conv3_%d'%i,reuse=reuse):
          conv3=residual_block(layers[-1],output_channel=512)
          layers.append(conv3)
       #assert conv3.get_shape().as_list()[1:]==[8,8,64]
       
       
    with tf.variable_scope('fc',reuse=reuse):
       in_channel=layers[-1].get_shape().as_list()[-1]
       bn_layer=batch_normalization_layer(layers[-1],in_channel)
       relu_layer=tf.nn.relu(bn_layer)
       global_pool=tf.reduce_mean(relu_layer,[1,2])
       
       #assert global_pool.get_shape().as_list()[-1:]==[64]
       global_pool=tf.nn.dropout(global_pool,keep_prob=pkeep)
       output=output_layers(global_pool,num_labels)
       layers.append(output)
       
       return layers[-1]
 
   
   
#%%

def loss(logits,labels):
   
   with tf.name_scope('Loss_op'):
      cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
         logits=logits,labels=labels)
   
      loss=tf.reduce_mean(cross_entropy)
      tf.summary.scalar('loss',loss)
      
      return loss
   

#%%
def accuracy(logits,labels):
   
   with tf.name_scope('Accuracy_op'):
        prediction=tf.nn.in_top_k(logits,labels,1)
        correct=tf.cast(prediction,tf.float32)
        correct=tf.reduce_mean(correct)
        tf.summary.scalar('accuracy',correct)
        return correct
     
#%%
def train_op(loss,learning_rate):
   with tf.name_scope('train_op'):
      train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
      return train_op
   
#%%
   
    
    
    
    
    
    
    
    
    
    

   
