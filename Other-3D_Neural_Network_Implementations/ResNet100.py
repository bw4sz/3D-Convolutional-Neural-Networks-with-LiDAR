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
lidars=np.load("H:\\Temp\\convnet_3d\\Leaf_On_Data\\unbuffered_leafon_quarterm.npy")
Ys=pd.read_csv("H:\\Temp\\convnet_3d\\Leaf_On_Data\\Ys.csv", sep=',')
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
    initial= tf.truncated_normal(shape, stddev=0.1)
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
    conv1=Conv_layer2(old_relu,[1,1,1],[1,1,1],shape,pad='SAME')
    lrn1=tf.contrib.layers.batch_norm(conv1, data_format='NHWC', center=True,scale=True, is_training=training)
    relued1=tf.nn.elu(lrn1)
    conv2=Conv_layer2(relued1,[2,2,2],[1,1,1],[shape[1],shape[1]],pad='SAME')
    lrn2=tf.contrib.layers.batch_norm(conv2, data_format='NHWC', center=True,scale=True, is_training=training)
    relued2=tf.nn.elu(lrn2)  
    conv3=Conv_layer2(relued2,[1,1,1],[1,1,1],[shape[1],shape[1]*4],pad='SAME')
    lrn3=tf.contrib.layers.batch_norm(conv3, data_format='NHWC', center=True,scale=True, is_training=training)
    added=old_relu+lrn3
    lyr_out=tf.nn.elu(added)    
    return (lyr_out)

def Res_layerB(old_relu, shape, dim_red=False):
    res_conv1=Conv_layer2(old_relu,[1,1,1],[2,2,2],[shape[1],shape[0]],pad='SAME')
    res_lrn1=tf.contrib.layers.batch_norm(res_conv1, data_format='NHWC', center=True,scale=True, is_training=training)
    res_relued1=tf.nn.relu(res_lrn1)
    res_conv2=Conv_layer2(res_relued1,[2,2,2],[1,1,1],[shape[0],shape[0]],pad='SAME')
    res_lrn2=tf.contrib.layers.batch_norm(res_conv2, data_format='NHWC', center=True,scale=True, is_training=training)
    res_relued2=tf.nn.relu(res_lrn2)  
    res_conv3=Conv_layer2(res_relued2,[1,1,1],[1,1,1],[shape[0],shape[2]],pad='SAME')
    res_lrn3=tf.contrib.layers.batch_norm(res_conv3, data_format='NHWC', center=True,scale=True, is_training=training)
    pth_conv=Conv_layer2(old_relu,[1,1,1],[2,2,2],[shape[1],shape[2]])
    pth_lrn=tf.contrib.layers.batch_norm(pth_conv, data_format='NHWC', center=True,scale=True, is_training=training)
    res_added=pth_lrn+res_lrn3
    res_lyr_out=tf.nn.relu(res_added)
    return (res_lyr_out)
        
#DEFINE MODEL   
#with tf.variable_scope("model", reuse=True): 
    ################initial conv layers######################
h_conv1=Conv_layer2(x_image,[2,2,3],[2,2,2],[1,64],pad='SAME')
h_lrn1=tf.contrib.layers.batch_norm(h_conv1, data_format='NHWC', center=True,scale=True, is_training=training)
h_relu1=tf.nn.relu(h_lrn1)
h_pool3=tf.nn.max_pool3d(h_relu1, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')

#FIRST RESNET LAYER
res_conv1=Conv_layer2(h_pool3,[1,1,1],[1,1,1],[64,64],pad='SAME')
res_lrn1=tf.contrib.layers.batch_norm(res_conv1, data_format='NHWC', center=True,scale=True, is_training=training)
res_relued1=tf.nn.elu(res_lrn1)
res_conv2=Conv_layer2(res_relued1,[2,2,2],[1,1,1],[64,64],pad='SAME')
res_lrn2=tf.contrib.layers.batch_norm(res_conv2, data_format='NHWC', center=True,scale=True, is_training=training)
res_relued2=tf.nn.elu(res_lrn2)  
res_conv3=Conv_layer2(res_relued2,[1,1,1],[1,1,1],[64,256],pad='SAME')
res_lrn3=tf.contrib.layers.batch_norm(res_conv3, data_format='NHWC', center=True,scale=True, is_training=training)
pth2_conv=Conv_layer2(h_pool3,[1,1,1],[1,1,1],[64,256])
pth2_lrn=tf.contrib.layers.batch_norm(pth2_conv, data_format='NHWC', center=True,scale=True, is_training=training)
res_added=pth2_lrn+res_lrn3
res_lyr_out=tf.nn.elu(res_added)
#Additional first level RESNET layers
resa1=Res_layer(res_lyr_out, [256,64])
resa2=Res_layer(resa1, [256,64])

#Second RESNET LAYER
res_lyr_out2=Res_layerB(resa2, [128,256,512])
#Additional second level RESNET layers
resb1=Res_layer(res_lyr_out2, [512,128])
resb2=Res_layer(resb1, [512,128])
resb3=Res_layer(resb2, [512,128])

#Third RESNET LAYER
res_lyr_out3=Res_layerB(resb3, [256,512,1024])
#Additional second level RESNET layers
resc1=Res_layer(res_lyr_out3, [1024,256])
resc2=Res_layer(resc1, [1024,256])
resc3=Res_layer(resc2, [1024,256])
resc4=Res_layer(resc3, [1024,256])
resc5=Res_layer(resc4, [1024,256])
resc6=Res_layer(resc5, [1024,256])
resc7=Res_layer(resc6, [1024,256])
resc8=Res_layer(resc7, [1024,256])
resc9=Res_layer(resc8, [1024,256])
resc10=Res_layer(resc9, [1024,256])
resc11=Res_layer(resc10, [1024,256])
resc12=Res_layer(resc11, [1024,256])
resc13=Res_layer(resc12, [1024,256])
resc14=Res_layer(resc13, [1024,256])
resc15=Res_layer(resc14, [1024,256])
resc16=Res_layer(resc15, [1024,256])
resc17=Res_layer(resc16, [1024,256])
resc18=Res_layer(resc17, [1024,256])
resc19=Res_layer(resc18, [1024,256])
resc20=Res_layer(resc19, [1024,256])
resc21=Res_layer(resc20, [1024,256])
resc22=Res_layer(resc21, [1024,256])

#Forth RESNET LAYER
res_lyr_out4=Res_layerB(resc22, [512,1024,2048])
resd2=Res_layer(res_lyr_out4, [2048,512])
resd3=Res_layer(resd2, [2048,512])

#fully connected layers
fc_pool=tf.nn.avg_pool3d(resd3, ksize=[1,3,3,7,1], strides=[1,3,3,7,1], padding='SAME')
shape = fc_pool.get_shape().as_list()
fc_flat=tf.reshape(fc_pool, [-1, shape[1] * shape[2] * shape[3]* shape[4]])

W_FCo=weight_variable([2048, 1000])
b_FCo=bias_variable([1])
relu_FCo=tf.nn.elu(tf.matmul(fc_flat,W_FCo) + b_FCo)

W_FCo2=weight_variable([1000, 1])
b_FCo2=bias_variable([1])
prediction=(tf.matmul(relu_FCo,W_FCo2) + b_FCo2)#output

#Compute loss
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)

###########################FEED IN DATA, ECT..####################
sess = tf.Session()
init=tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

indices=np.load('H:\\Temp\\convnet_3d\\Leaf_On_Data\\withheld.npy')
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

training_summary = tf.summary.scalar("training_accuracy", loss)
validation_summary = tf.summary.scalar("validation_accuracy", loss)
summary_writer = tf.summary.FileWriter('data/home/eayrey/Python/Convnet_3d/resnet/Summaries/Summary')

fd = open('H:\\Temp\\convnet_3d\\resnet\\summarized_batch30E.csv','a')
record_low=100000000
overtrain_indicator=0
for i in range(50000):
    indices=np.random.randint(low=0, high=len(train_ys), size=[15,])
    batch_xs=train_xs[indices]
    batch_ys=train_ys[indices]
    #batch_xs, batch_ys=mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i%100==0:
        #record training accuracy
        train_acc, train_summ = sess.run(
                [loss, training_summary], 
                feed_dict={xs: batch_xs, ys: batch_ys})
        summary_writer.add_summary(train_summ, i) 
        #record validation accuracy
        
        valid_accuracies=[]
        for n in range(40):
            batch_valid_xs=validation_xs[25*n:25*(n+1)]
            batch_valid_ys=validation_ys[25*n:25*(n+1)]
            
            valid_acc, valid_summ = sess.run(
                [loss, validation_summary],
                feed_dict={xs: batch_valid_xs, ys: batch_valid_ys})
            if i==0:
                valid_summaries=valid_summ
                valid_accuracies.append(valid_acc)
            if i>0:
                valid_summaries=tf.summary.merge([valid_summaries,valid_summ])
                valid_accuracies.append(valid_acc)
        valid_acc2=np.mean(valid_accuracies)    
        #summary_writer.add_summary(valid_summaries, i)  
        fd.write(str(i)+','+str(train_acc)+','+str(valid_acc2)+'\n')
        #save model if its the best so far

            
	  #finish model training if we've done more than 1000 steps and the validation data is consistantly underperforming training    
        if i>500:
            if valid_acc2 < record_low:
                record_low=valid_acc2
                saver.save(sess, 'H:\\Temp\\convnet_3d\\resnet\\resnet_30_0001_'+str(train_acc)+'_'+str(valid_acc2), global_step=i)
            if float(valid_acc2)/train_acc > 1.2:
                overtrain_indicator+=1
            else:
                overtrain_indicator=0
            if overtrain_indicator == 10:
                    break
        print(i, train_acc, valid_acc2, overtrain_indicator) 
print ('Done! Record low: '+str(record_low)) 
fd.close()
#quit()
