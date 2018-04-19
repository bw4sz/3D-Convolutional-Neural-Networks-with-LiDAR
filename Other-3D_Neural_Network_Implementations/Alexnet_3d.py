# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:11:35 2017

@author: Elias Ayrey
"""



from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
#import pandas
#from tensorflow.examples.tutorials.mnist import input_data
#lidars=np.load("H:\\Temp\\convnet_3d\\Leaf_Off_Data\\unbuffered_leafoff_quarterm.npy")
#Ys=pd.read_csv("H:\\Temp\\convnet_3d\\Leaf_Off_Data\\Ys.csv", sep=',')

lidars=np.load("H:/Temp/convnet_3d/unbuffered_lidars_quarterm.npy")
Ys=pd.read_csv("H:\\Temp\\convnet_3d\\Ys.csv", sep=',')
Ys=Ys['Biomass_AG']
Ys=np.asarray(Ys)
#Effects=pd.read_csv("H:\\Temp\\convnet_3d\\Effects2.csv", sep=',')
#Effects=Effects[['Aspect','cell_ppm','Slope','TRI']]
#Effects=np.asarray(Effects)


#normalize, organize
#Effects[:,0]=Effects[:,0]-np.mean(Effects[:,0])
#Effects[:,0]=Effects[:,0]/np.std(Effects[:,0])
#Effects[:,1]=Effects[:,1]-np.mean(Effects[:,1])
#Effects[:,1]=Effects[:,1]/np.std(Effects[:,1])
#Effects[:,2]=Effects[:,2]-np.mean(Effects[:,2])
#Effects[:,2]=Effects[:,2]/np.std(Effects[:,2])
#Effects[:,3]=Effects[:,3]-np.mean(Effects[:,3])
#Effects[:,3]=Effects[:,3]/np.std(Effects[:,3])
lidars=lidars.astype('float32')
Ys=Ys.astype('float32')
Ys=Ys.reshape(-1,1)
#Effects=Effects.astype('float32')

#define placeholders for model
xs=tf.placeholder(tf.float32, [None,40,40,105], name='Xinput') 
ys=tf.placeholder(tf.float32, [None,1], name='Yinput') 
#effect=tf.placeholder(tf.float32, [None,4], name='effect') 

#Define keep probability for dropout layers
keep_prob = tf.constant(0.5)
#the minus 1 has to do with the number of images, the 1 has to do with 'bands'
x_image=tf.reshape(xs, [-1,40,40,105,1])

#defining weight variables from random normal curve with a shape of the input
def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#defining bias variables as all starting as .1, with shape of the input    
def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
   
#3d normalization routine, specify if we're training or testing model!!!!!
training = tf.placeholder_with_default(True, shape=())

#DEFINE MODEL    
################first conv layer######################
#first the filter is defined randomly, not predefined

W_conv1=weight_variable([2,2,2,1,96])#filter size 2x2x2, 1 band, 96 convolutions out
b_conv1=bias_variable([96])#biases for each of the 96 convolutions
conv1= tf.nn.conv3d(x_image, W_conv1, strides=[1,2,2,2,1], padding='VALID')#convolutions, stride equals 2, no padding
#1 in begining is fixed unless youre crazy, one in end refers to bands, and the middle refer to xy. [1,x,y,z,1]

added = tf.nn.bias_add(conv1, b_conv1)
#add bias to weight*inputs

h_conv1=tf.nn.relu(added)#activator
#Relu

h_norm1=tf.contrib.layers.batch_norm(h_conv1, data_format='NHWC', center=True,scale=True, is_training=training,scope='cnn3d-batch_norm')
#Normalize data

h_pool1=tf.nn.max_pool3d(h_conv1, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='VALID')#again uses a window and takes max with strides
#h_pool1 is now half the size, 
#Now 19x19x50 with 96 convolutions


################second conv layer######################
W_conv2=weight_variable([2,2,2,96,256])#filter size 5x5, 1 band, 32 convolutions out
b_conv2=bias_variable([256])#biases for each of the 32 convolutions
conv2= tf.nn.conv3d(h_pool1, W_conv2, strides=[1,2,2,2,1], padding='VALID')#convolutions, stride equals 1, 
#1 in begining is fixed unless your crazy, one in end refers to bands, and the middle refer to xy. [1,x,y,1]
added = tf.nn.bias_add(conv2, b_conv2)
#add bias to weight*inputs
h_conv2=tf.nn.relu(added)#activator
#Relu
h_norm2=tf.contrib.layers.batch_norm(h_conv2, data_format='NHWC', center=True,scale=True, is_training=training,scope='cnn3d-batch_norm2')
#Normalize data
h_pool2=tf.nn.max_pool3d(h_conv2, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='VALID')#again uses a window and takes max with strides
#h_pool1 is now half the size, 

################third conv layer######################
W_conv3=weight_variable([2,2,2,256,384])#filter size 5x5, 1 band, 32 convolutions out
b_conv3=bias_variable([384])#biases for each of the 32 convolutions
conv3= tf.nn.conv3d(h_pool2, W_conv3, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
#1 in begining is fixed unless your crazy, one in end refers to bands, and the middle refer to xy. [1,x,y,1]
added = tf.nn.bias_add(conv3, b_conv3)
#add bias to weight*inputs
h_conv3=tf.nn.relu(added)#activator

################forth conv layer######################
W_conv4=weight_variable([2,2,2,384,384])#filter size 5x5, 1 band, 32 convolutions out
b_conv4=bias_variable([384])#biases for each of the 32 convolutions
conv4= tf.nn.conv3d(h_conv3, W_conv4, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
#1 in begining is fixed unless your crazy, one in end refers to bands, and the middle refer to xy. [1,x,y,1]
added = tf.nn.bias_add(conv4, b_conv4)
#add bias to weight*inputs
h_conv4=tf.nn.relu(added)#activator  

################fifth conv layer######################
W_conv4=weight_variable([2,2,2,384,256])#filter size 5x5, 1 band, 32 convolutions out
b_conv4=bias_variable([256])#biases for each of the 32 convolutions
conv4= tf.nn.conv3d(h_conv3, W_conv4, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
#1 in begining is fixed unless your crazy, one in end refers to bands, and the middle refer to xy. [1,x,y,1]
added = tf.nn.bias_add(conv4, b_conv4)
#add bias to weight*inputs
h_conv4=tf.nn.relu(added)#activator  

h_pool4=tf.nn.max_pool3d(h_conv2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')#again uses a window and takes max with strides
#unclear if this pooling layer should have a stride of 1 or 2, sources differ

#We need to flatten the thing to feed into a normal layer, also add effects
shape = h_pool4.get_shape().as_list()
hpool4_flat=tf.reshape(h_pool4, [-1, shape[1] * shape[2] * shape[3]* shape[4]])
#reg_inputs=tf.concat(axis=1, values=[hpool4_flat,effect])
shape2 = hpool4_flat.get_shape().as_list()

#Error checking snipits
#sess = tf.Session()
#init=tf.global_variables_initializer()       
#sess.run(init)
#grr=sess.run(W_l1.get_shape())
#print(grr)
##############regular layers#########################    
W_l1=weight_variable([shape2[1],4096])#input is size of flattened thing, The 4 refers to effects like ppm.
b_l1=bias_variable([4096])
h_l1=tf.nn.relu(tf.matmul(hpool4_flat,W_l1) + b_l1)
h_l1_drop=tf.nn.dropout(h_l1 , keep_prob)

W_l2=weight_variable([4096, 4096])
b_l2=bias_variable([4096])
h_l2=tf.nn.relu(tf.matmul(h_l1_drop,W_l2) + b_l2)
h_l2_drop=tf.nn.dropout(h_l2 , keep_prob)

W_l3=weight_variable([4096, 1000])
b_l3=bias_variable([1000])
h_l3=tf.nn.relu(tf.matmul(h_l2_drop,W_l3) + b_l3)
h_l3_drop=tf.nn.dropout(h_l3 , keep_prob)

W_l4=weight_variable([1000, 1])
b_l4=bias_variable([1])
prediction=(tf.matmul(h_l3_drop,W_l4) + b_l4)#output

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

#####################################################
#CUDA_VISIBLE_DEVICES=""
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
#config.gpu_options.allow_growth=True
#config = tf.ConfigProto()
#config.gpu_options.allocator_type = 'BFC'
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
sess = tf.Session()
#sess = tf.Session()#config=config)
init=tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

#Seperate test and validation data
indices=np.random.randint(low=0, high=len(Ys), size=[2000,])
withheld_xs=lidars[indices]
test_xs=withheld_xs[0:1000]
validation_xs=withheld_xs[1000::]

withheld_ys=Ys[indices]
test_ys=withheld_ys[0:1000]
validation_ys=withheld_ys[1000::]

#test_batch_Es=Effects[indices]
#withheld_effects=Effects[indices]
#test_effects=withheld_effects[0:1000]
#validation_effects=withheld_effects[1000::]

mask = np.ones(len(Ys), np.bool)
mask[indices] = 0
train_xs = lidars[mask]
train_ys = Ys[mask]
#train_Es=Effects[mask]

training_summary = tf.summary.scalar("training_accuracy", loss)
validation_summary = tf.summary.scalar("validation_accuracy", loss)
summary_writer = tf.summary.FileWriter('H:Temp//Model_saves//summary')

fd = open('H:/Temp/convnet_3d/New folder/summarized.csv','a')
record_low=100000000
overtrain_indicator=0
for i in range(50000):
    indices=np.random.randint(low=0, high=len(train_ys), size=[25,])
    batch_xs=train_xs[indices]
    batch_ys=train_ys[indices]
#    batch_es=train_Es[indices]
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
#            batch_valid_es=validation_effects[25*n:25*(n+1)]
            
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
        if valid_acc2 < record_low:
            record_low=valid_acc2
        fd.write(str(i)+','+str(train_acc)+','+str(valid_acc2)+'\n')
        #save model if its the best so far

            
	  #finish model training if we've done more than 1000 steps and the validation data is consistantly underperforming training    
        if i>1000:
            if valid_acc2 < record_low:
                record_low=valid_acc2
                saver.save(sess, 'H:\\Temp\\Model_saves\\Alexnet_'+str(train_acc)+'_'+str(valid_acc2), global_step=i) 
            if float(valid_acc)/train_acc > 1.2:
                overtrain_indicator+=1
            else:
                overtrain_indicator=0
            if overtrain_indicator == 10:
                    break
        print(i, train_acc, valid_acc2, overtrain_indicator) 
print ('Done! Record low: '+str(record_low)) 
fd.close()
