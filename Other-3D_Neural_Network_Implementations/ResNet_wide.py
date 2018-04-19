# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:37:34 2017

@author: Elias Ayrey
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
#import pandas
#from tensorflow.examples.tutorials.mnist import input_data
lidars=np.load("/data/home/eayrey/Python/Convnet_3d/Leaf_On/unbuffered_leafon_quarterm.npy")
Ys=pd.read_csv("/data/home/eayrey/Python/Convnet_3d/Leaf_On/Ys.csv", sep=',')
Ys=Ys['Biomass_AG']
Ys=np.asarray(Ys)

#normalize, organize
lidars=lidars.astype('float32')
Ys=Ys.astype('float32')
Ys=Ys.reshape(-1,1)

xs=tf.placeholder(tf.float32, [None,40,40,105], name='Xinput') 
ys=tf.placeholder(tf.float32, [None,1], name='Yinput') 

#Define keep probability for dropout layers
keep_prob = tf.constant(0.5)
#the minus 1 has to do with the number of images, the 1 has to do with 'bands'
x_image=tf.reshape(xs, [-1,40,40,105,1])

#3d normalization routine, specify if we're training or testing model!!!!!
training = tf.placeholder_with_default(True, shape=())

#defining weight variables from random normal curve with a shape of the input
def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

#defining bias variables as all starting as .1, with shape of the input    
def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def Conv_layer2(inputs,kernal,stride,shape, pad='VALID'):
    weights=weight_variable(kernal+shape)
    biases=bias_variable([shape[1]])
    conv=tf.nn.conv3d(inputs, weights, strides=[1]+stride+[1], padding=pad)
    added =tf.nn.bias_add(conv, biases)
    return added 

def Res_layer(old_relu, shape, dim_red=False):
    if dim_red==True:
        #thread one
        conv1=Conv_layer2(old_relu,[2,2,2],[2,2,2],[shape[0],shape[1]],pad='SAME')
        lrn1=tf.contrib.layers.batch_norm(conv1, data_format='NHWC', center=True,scale=True, is_training=training)
        relu1=tf.nn.elu(lrn1)
        conv2=Conv_layer2(relu1,[2,2,2],[1,1,1],[shape[1],shape[1]],pad='SAME')
        #thread two
        res_conv1=Conv_layer2(old_relu,[1,1,1],[2,2,2],[shape[0],shape[1]],pad='SAME')
        res_added=res_conv1+conv2
        return res_added
    else:
        conv1=Conv_layer2(old_relu,[2,2,2],[1,1,1],[shape[0],shape[1]],pad='SAME')
        lrn1=tf.contrib.layers.batch_norm(conv1, data_format='NHWC', center=True,scale=True, is_training=training)
        relu1=tf.nn.elu(lrn1)
        conv2=Conv_layer2(relu1,[2,2,2],[1,1,1],[shape[1],shape[1]],pad='SAME')
        res_added=old_relu+conv2
        return res_added

    
        
#DEFINE MODEL   
#with tf.variable_scope("model", reuse=True): 
    ################initial conv layers######################
h_conv1=Conv_layer2(x_image,[2,2,3],[1,1,1],[1,16],pad='SAME')
h_lrn1=tf.contrib.layers.batch_norm(h_conv1, data_format='NHWC', center=True,scale=True, is_training=training)
h_relu1=tf.nn.elu(h_lrn1)
#h_pool3=tf.nn.max_pool3d(h_relu1, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')

#FIRST RESNET LAYER
conv1a=Conv_layer2(h_relu1,[2,2,2],[2,2,2],[16,32],pad='SAME')
lrn1a=tf.contrib.layers.batch_norm(conv1a, data_format='NHWC', center=True,scale=True, is_training=training)
relu1a=tf.nn.elu(lrn1a)
conv2a=Conv_layer2(relu1a,[2,2,2],[1,1,1],[32,32],pad='SAME')
#thread two
res_conv1=Conv_layer2(h_relu1,[1,1,1],[2,2,2],[16,32],pad='SAME')
res_added=res_conv1+conv2a

#Additional first level RESNET layers
resa1=Res_layer(res_added, [32,32])

#Second RESNET LAYER
resb1=Res_layer(resa1, [32,64], dim_red=True)
resb2=Res_layer(resb1, [64,64])
resb3=Res_layer(resb2, [64,64])
resb4=Res_layer(resb3, [64,64])

#Third RESNET LAYER
resc1=Res_layer(resb4, [64,128], dim_red=True)
resc2=Res_layer(resc1, [128,128])

#Forth RESNET LAYER
resd1=Res_layer(resc2, [128,256], dim_red=True)
resd2=Res_layer(resd1, [256,256])

#fully connected layers
lrn2a=tf.contrib.layers.batch_norm(resd2, data_format='NHWC', center=True,scale=True, is_training=training)
relu2a=tf.nn.elu(lrn2a)
fc_pool=tf.nn.avg_pool3d(relu2a, ksize=[1,3,3,7,1], strides=[1,3,3,7,1], padding='SAME')
shape = fc_pool.get_shape().as_list()
fc_flat=tf.reshape(fc_pool, [-1, shape[1] * shape[2] * shape[3]* shape[4]])

W_FCo2=weight_variable([256, 1])
b_FCo2=bias_variable([1])
prediction=(tf.matmul(fc_flat,W_FCo2) + b_FCo2)#output

#Compute loss
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(.00005).minimize(loss)

###########################FEED IN DATA, ECT..####################
sess = tf.Session()
init=tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

indices=np.load('/data/home/eayrey/Python/Convnet_3d/Leaf_On/withheld.npy')
withheld_xs=lidars[indices]
test_xs=withheld_xs[0:1000]
validation_xs=withheld_xs[1000::]

withheld_ys=Ys[indices]
test_ys=withheld_ys[0:1000]
validation_ys=withheld_ys[1000::]

mask = np.ones(len(Ys), np.bool)
mask[indices] = 0
train_xs = lidars[mask]
train_ys = Ys[mask]

fd = open('/data/home/eayrey/Python/Convnet_3d/Leaf_On/Resnet_Wide/summarized.csv','a')
record_low=100000000
overtrain_indicator=0
for i in range(500000):
    indices=np.random.randint(low=0, high=len(train_ys), size=[20,])
    batch_xs=train_xs[indices]
    batch_ys=train_ys[indices]
    #batch_xs, batch_ys=mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i%100==0:
        #record training accuracy
        train_acc= sess.run(
                loss, 
                feed_dict={xs: batch_xs, ys: batch_ys})
        
        valid_accuracies=[]
        for n in range(40):
            batch_valid_xs=validation_xs[25*n:25*(n+1)]
            batch_valid_ys=validation_ys[25*n:25*(n+1)]
            
            valid_acc= sess.run(
                loss,
                feed_dict={xs: batch_valid_xs, ys: batch_valid_ys})
            if i==0:
                valid_accuracies.append(valid_acc)
            if i>0:
                valid_accuracies.append(valid_acc)
        valid_acc2=np.mean(valid_accuracies)    
        #summary_writer.add_summary(valid_summaries, i)  
        fd.write(str(i)+','+str(train_acc)+','+str(valid_acc2)+'\n')
        #save model if its the best so far

            
	  #finish model training if we've done more than 1000 steps and the validation data is consistantly underperforming training    
        if i>500:
            if valid_acc2 < record_low:
                record_low=valid_acc2
                saver.save(sess, '/data/home/eayrey/Python/Convnet_3d/Leaf_On/Resnet_Wide/Resnet_wide_'+str(train_acc)+'_'+str(valid_acc2), global_step=i)
            if float(valid_acc2)/train_acc > 2:
                overtrain_indicator+=1
            else:
                overtrain_indicator=0
            if overtrain_indicator == 10000:
                    break
        if i%3000==0:
            fc = open('/data/home/eayrey/Python/Convnet_3d/Leaf_On/Resnet_Wide/Summary/Resnet_wide'+str(train_acc)+'_'+str(valid_acc2)+'_'+str(20)+'_'+str(i)+'.csv','a')
            fc.close()
        print(i, train_acc, valid_acc2, overtrain_indicator) 
print ('Done! Record low: '+str(record_low)) 
fd.close()
quit()
