# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:47:51 2017

@author: Elias Ayrey
"""



from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow as tf
import numpy as np
import pandas as pd

#Load array of voxelized point cloud grid cells developed using the point_cloud_voxelizer.py
lidars=np.load("H:\\Temp\\convnet_3d\\Leaf_On_Data\\unbuffered_leafon_quarterm.npy")
#Load dependant variable and reformat to a (-1,1) array
Ys=pd.read_csv("H:\\Temp\\convnet_3d\\Leaf_On_Data\\Ys.csv", sep=',')
Ys=Ys['Biomass_AG']
Ys=np.asarray(Ys)
Ys=Ys.astype('float32')
Ys=Ys.reshape(-1,1)

#defining weight variables from random normal curve with a shape of the input
def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#defining bias variables as all starting as .1, with shape of the input    
def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
   
################   DEFINE MODEL   ############################
#specify placeholders for tensorflow in the shape of our input data
xs=tf.placeholder(tf.float32, [None,40,40,105], name='Xinput') 
ys=tf.placeholder(tf.float32, [None,1], name='Yinput') 
#training switch, used with batch norm. Switch to false during validation.
training = tf.placeholder_with_default(True, shape=())
keep_prob = tf.constant(0.5)

#Our model input reshaped into tensorflow's prefered standard
#the minus 1 has to do with the number of plots
x_image=tf.reshape(xs, [-1,40,40,105,1])

################ first conv layer ######################
W_conv1=weight_variable([6,6,16,1,6])#filter size 5x5, 1 band, 32 convolutions out
b_conv1=bias_variable([6])#biases for each of the 32 convolutions
conv1= tf.nn.conv3d(x_image, W_conv1, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
#1 in begining is fixed unless your crazy, one in end refers to bands, and the middle refer to xy. [1,x,y,z,1]
added = tf.nn.bias_add(conv1, b_conv1)
h_conv1=tf.nn.relu(added)#activator
h_norm1=tf.layers.batch_normalization(h_conv1, data_format='NHWC', center=True,scale=True, is_training=training,scope='cnn3d-batch_norm')
h_pool1=tf.nn.max_pool3d(h_conv1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')#again uses a window and takes max with strides

################ second conv layer ######################
W_conv2=weight_variable([6,6,16,6,16])#filter size 5x5, 1 band, 32 convolutions out
b_conv2=bias_variable([16])#biases for each of the 32 convolutions
conv2= tf.nn.conv3d(h_pool1, W_conv2, strides=[1,1,1,1,1], padding='VALID')
added = tf.nn.bias_add(conv2, b_conv2)
h_conv2=tf.nn.relu(added)#activator
h_norm2=tf.layers.batch_normalization(h_conv2, data_format='NHWC', center=True,scale=True, is_training=training,scope='cnn3d-batch_norm2')
h_pool2=tf.nn.max_pool3d(h_conv2, ksize=[1,2,2,3,1], strides=[1,2,2,3,1], padding='VALID')#again uses a window and takes max with strides

############## Fully Connected Layers #########################    
shape = h_pool2.get_shape().as_list()
hpool2_flat=tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]* shape[4]])
reg_inputs=tf.concat(axis=1, values=[hpool2_flat,effect])
shape2 = reg_inputs.get_shape().as_list()

#Pools into a fully connected layer with 2048 neurons
W_l1=weight_variable([shape2[1],2048])
b_l1=bias_variable([2048])
h_l1=tf.nn.relu(tf.matmul(reg_inputs,W_l1) + b_l1)
h_l1_drop=tf.nn.dropout(h_l1 , keep_prob)

#Pools into a fully connected layer with 512 neurons
W_l2=weight_variable([2048, 512])
b_l2=bias_variable([512])
h_l2=tf.nn.relu(tf.matmul(h_l1_drop,W_l2) + b_l2)
h_l2_drop=tf.nn.dropout(h_l2 , keep_prob)

#Final output layer with prediction
W_l4=weight_variable([512, 1])
b_l4=bias_variable([1])
prediction=(tf.matmul(h_l2_drop,W_l4) + b_l4)#output

#Loss (MSE)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
#Necessary crap for batch normalization, cuz tensorflow
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    #specify step size here, larger step size may work better
    train_step=tf.train.AdamOptimizer(0.00001).minimize(loss)

###########################################################################################################
########################### INITIALIZE MODEL #################################
sess = tf.Session()
init=tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2)
sess.run(init)

############################ SEGMENT DATA ######################################
#Seperate test and validation data
indices=np.random.randint(low=0, high=len(Ys), size=[2000,])
withheld_xs=lidars[indices]
test_xs=withheld_xs[0:1000]
validation_xs=withheld_xs[1000::]

withheld_ys=Ys[indices]
test_ys=withheld_ys[0:1000]
validation_ys=withheld_ys[1000::]

withheld_effects=Effects[indices]
test_effects=withheld_effects[0:1000]
validation_effects=withheld_effects[1000::]

mask = np.ones(len(Ys), np.bool)
mask[indices] = 0
train_xs = lidars[mask]
train_ys = Ys[mask]

#tensorflow's offical summary file. Pshhh
#summary_writer = tf.summary.FileWriter('H:/Temp/convnet_3d/Lenet/Model_saves/Summary')

############################ TRAINING ######################################
batchS=25
record_low=12333333333
#summary file
fd = open('H:/Temp/convnet_3d/InceptionV3/summarized.csv','a')
overtrain_indicator=0
for i in range(1500000):
    #for each trainding step, withhold a batch of data, train 
    indices=np.random.randint(low=0, high=len(train_ys), size=[batchS,])
    batch_xs=train_xs[indices]
    batch_ys=train_ys[indices]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, training: True})
    
    #run a validation test every 50 steps
    if i%50==0:
        train_acc= sess.run(loss, feed_dict={xs: batch_xs, ys: batch_ys,es:batch_es, training: True})
        
        #assess model accuracy on validation data in small peices (don't want to overload GPU mem)
        valid_accuracies=[]
        RMSEs=[]
        for n in range(20):
            batch_valid_xs=validation_xs[50*n:50*(n+1)]
            batch_valid_ys=validation_ys[50*n:50*(n+1)]
            
            #obtain prediction and loss (MSE) from model, training set to false
            [valid_acc, pred]= sess.run([loss, prediction] ,feed_dict={xs: batch_valid_xs, ys: batch_valid_ys, training: False})
            
            valid_accuracies.append(valid_acc)
            
            #calculate RMSE  
            RMSE= np.sqrt(np.mean((np.asarray(batch_valid_ys).flatten()-np.asarray(pred).flatten())**2))
            RMSEs.append(RMSE) 
        RMSE=np.mean(RMSEs)     
        valid_acc2=np.mean(valid_accuracies)    
            
        if i>1000:
            if valid_acc2 < record_low:
                record_low=valid_acc2
                #Save model somewhere
                saver.save(sess, 'H:\\Temp\\Chapter_2_bits\\Model_Testing\\Old_Model\\Lenet_Old_05added_2000-1000-250-10-250-1000'+str(train_acc)+'_'+str(valid_acc2)+'_'+str(RMSE), global_step=i)
                overtrain_indicator=0
            else:
                overtrain_indicator+=1
                #if the model is no longer improving, stop training.
#                if overtrain_indicator == 5000:
#                        break
        print(i, train_acc, valid_acc2, RMSE, overtrain_indicator, record_low)   
print ('Done! Record low: '+str(record_low))  
sess.close()
tf.reset_default_graph()
fd.close()
#quit()