import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf

#%%
TF_VERSION=float('.'.join(tf.__version__.split('.')[:2]))   #check tf version

#%%
class DenseNet(object):
    
    def __init__(self,images,labels,n_classes,growth_rate,depth,
                 total_blocks,keep_prob,learning_rate,
                 weight_decay,nesterov_momentum,model_type,
                 reduction=1.0,
                 bc_mode=False,
                 is_training=True):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf
        Args:
            image_shape:  tensor 4D
            n_classes:l  labels classes
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.images=images
        self.labels=labels
        # self.image_shape=[int(x) for x in images.get_shape()]
        self.n_classes=n_classes
        self.depth=depth
        self.growth_rate=growth_rate
        self.learning_rate=learning_rate
        self.is_training=is_training
        
        #how many features will be receiced after first convolution
        #value the same as in the original Torch code
        
        self.first_output_features=growth_rate*2
        self.total_blocks=total_blocks
        self.layers_per_block=(depth-(total_blocks+1))//total_blocks
        self.bc_mode=bc_mode
        
        #compression rate at the transition layers
        self.reduction=reduction
        
        if not bc_mode:
            print("Build %s model with %d blocks,%d composite layers each."%(
                    model_type,self.total_blocks,self.layers_per_block))
            
        if bc_mode:
            self.layers_per_block=self.layers_per_block//2
            print("Build %s model with %d blocks,%d bottleneck layers and %d composite layers each"%(
                    model_type,self.total_blocks,self.layers_per_block,self.layers_per_block))
            
        
        print('Reduction at transition layers:%.1f'%(self.reduction))
        
        self.keep_prob=keep_prob
        self.weight_decay=weight_decay
        self.nesterov_momentum=nesterov_momentum
        self.model_type=model_type
        self.batches_step=0
        
    def composite_function(self,X,out_channel,kernel_size=3):
        """
        Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        
        with tf.variable_scope('composite_function'):
           
           #BN
           output=self.batch_norm(X)
           
           #ReLU
           output=tf.nn.relu(output)
           
           #convolution
           output=self.conv2d(
                 output,out_channel=out_channel,kernel_size=kernel_size)
           
           #Dropout
           output=self.dropout(output)
           
        return output
      
      
    def bottleneck(self,X,out_channel):
       
       with tf.variable_scope('bottleneck'):
          output=self.batch_norm(X)
          output=tf.nn.relu(output)
          inter_features=out_channel*4
          
          output=self.conv2d(
                output,out_channel=inter_features,kernel_size=1,padding='VALID')
          
          output=self.dropout(output)
       return output
    
   
    def add_internal_layer(self,X,growth_rate):
        """
        Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        if not self.bc_mode:
           comp_out=self.composite_function(X,out_channel=growth_rate,
                                            kernel_size=3)
           
        elif self.bc_mode:
           bottleneck_out=self.bottleneck(X,growth_rate)
           comp_out=self.composite_function(
                 bottleneck_out,out_channel=growth_rate,kernel_size=3)
           
        output=tf.concat([X,comp_out],3)
        return output
     
      
    def add_block(self,X,growth_rate,layers_per_block):
        """
        Add N H_1 internal layers
        """
        output=X
        for layer in range(layers_per_block):
           with tf.variable_scope("layer_%d"%layer):
              output=self.add_internal_layer(output,growth_rate)
              
        return output
     
      
      
    def transition_layer(self,X):
       """
       Call H_1 composite function with 1X1 kernel and afteraverage pooling
       """
       out_features=int(int(X.get_shape()[-1])*self.reduction)
       output=self.composite_function(
             X,out_channel=out_features,kernel_size=1)
       
       output=self.avg_pool(output,k=2)
       return output
    
      
    def transition_layer_to_classes(self,X):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        
        #BN
        output=self.batch_norm(X)
        
        #ReLU
        output=tf.nn.relu(output)
        
        #average pooling
        last_pool_kernel=int(output.get_shape()[-2])
        output=self.avg_pool(output,k=last_pool_kernel)
        
        #FC
        features_total=int(output.get_shape()[-1])
        output=tf.reshape(output,[-1,features_total])
        
        W=self.weight_variable_xavier(
              [features_total,self.n_classes],name='W')
        
        bias=self.bias_variable([self.n_classes])
        
        logits=tf.matmul(output,W)+bias
        return logits
     
    def conv2d(self, X, out_channel, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
       
        in_features = int(X.get_shape()[-1])
        
        #kernel generate  
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_channel],
            name='kernel')
        output = tf.nn.conv2d(X, kernel, strides, padding)
        return output

    def avg_pool(self,X, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(X, ksize, strides, padding)
        return output

    def batch_norm(self, X):
        output = tf.contrib.layers.batch_norm(
            X, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self,X):
        output=tf.nn.dropout(X,self.keep_prob)
        return output
     
    def weight_variable_msra(self, shape, name):
        
       return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)
      
      
    def _build_graph(self):
       growth_rate=self.growth_rate
       layers_per_block=self.layers_per_block
       
       #first-initial 3x3 conv to first_output_features
       with tf.variable_scope('Initial_convolution'):
          output=self.conv2d(
                self.images,
                out_channel=self.first_output_features,
                kernel_size=3)
          
       #add N required blocks
       for block in range(self.total_blocks):
          
          with tf.variable_scope('Block_%d'%block):
             output=self.add_block(output,growth_rate,layers_per_block)
         
          if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output) 
             
       with tf.variable_scope('Transition_to_claases'):
          logits=self.transition_layer_to_classes(output)
      
       prediction=tf.nn.softmax(logits)
       
       
       #Losses
       with tf.name_scope('Loss'):
          cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
             logits=logits,labels=self.labels))
          tf.summary.scalar('loss',cross_entropy)
          self.cross_entropy=cross_entropy
       
       l2_loss=tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
       
       #optimizer and train step
       optimizer=tf.train.MomentumOptimizer(self.learning_rate,
                                            self.nesterov_momentum,use_nesterov=True)
       
       self.train_step=optimizer.minimize(cross_entropy+l2_loss*self.weight_decay)
       
       with tf.name_scope('Accuracy'):
          correct_prediction=tf.nn.in_top_k(prediction,self.labels,1)
          correct=tf.cast(correct_prediction,tf.float32)
          tf.summary.scalar('accuracy',tf.reduce_mean(correct))
          self.accuracy=tf.reduce_mean(correct)
#%%      
