# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:06:58 2017

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
lidars=np.load("H:\\Temp\\convnet_3d\\Leaf_On_Data\\unbuffered_leafon_quarterm.npy")
Ys=pd.read_csv("H:\\Temp\\convnet_3d\\Leaf_On_Data\\Ys.csv", sep=',')
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
#lidars=lidars.astype('float32')
Ys=Ys.astype('float32')
Ys=Ys.reshape(-1,1)
#Effects=Effects.astype('float32')


#defining weight variables from random normal curve with a shape of the input
def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=.05)
    return tf.Variable(initial)

#defining bias variables as all starting as .1, with shape of the input    
def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def Conv_layer(inputs,kernal,stride,shape, pad='VALID'):
    weights=weight_variable(kernal+shape)
    biases=bias_variable([shape[1]])
    conv=tf.nn.conv3d(inputs, weights, strides=[1]+stride+[1], padding=pad)
    added =tf.nn.bias_add(conv, biases)
    norm=tf.contrib.layers.batch_norm(added, data_format='NHWC', center=True,scale=True, is_training=training)
    h_conv=tf.nn.relu(norm)
    return h_conv

def Inception_layer1(intensor, outshapes):
    shape = intensor.get_shape().as_list()
    #first bit
    Ia_pool1=tf.nn.max_pool3d(intensor, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
    Ia_conv1=Conv_layer(Ia_pool1,[1,1,1],[1,1,1],[shape[4],outshapes[3]],pad='SAME')
    #second bit
    Ib_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],64],pad='SAME')
    Ib_conv2=Conv_layer(Ib_conv1,[2,2,3],[1,1,1],[64,96],pad='SAME')
    Ib_conv3=Conv_layer(Ib_conv2,[2,2,3],[1,1,1],[96,outshapes[2]],pad='SAME')
    #third bit
    Ic_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],48],pad='SAME')
    Ic_conv2=Conv_layer(Ic_conv1,[3,3,4],[1,1,1],[48,outshapes[1]],pad='SAME')
    #forth bit
    Id_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],outshapes[0]],pad='SAME')
    
    I_concat=tf.concat(axis=4, values=[Ia_conv1,Ib_conv3,Ic_conv2,Id_conv1])
    return I_concat

def Inception_layer2(intensor, outshapes, x):   
    shape = intensor.get_shape().as_list()
    #first bit
    Ia_pool1=tf.nn.max_pool3d(intensor, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
    Ia_conv1=Conv_layer(Ia_pool1,[1,1,1],[1,1,1],[shape[4],outshapes[3]],pad='SAME')
    #second bit
    Ib_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],x],pad='SAME')
    Ib_conv2=Conv_layer(Ib_conv1,[1,1,6],[1,1,1],[x,x],pad='SAME')
    Ib_conv3=Conv_layer(Ib_conv2,[1,5,1],[1,1,1],[x,x],pad='SAME')
    Ib_conv4=Conv_layer(Ib_conv3,[5,1,1],[1,1,1],[x,x],pad='SAME')
    Ib_conv5=Conv_layer(Ib_conv4,[1,1,6],[1,1,1],[x,outshapes[2]],pad='SAME')
    #third bit
    Ic_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],x],pad='SAME')
    Ic_conv2=Conv_layer(Ic_conv1,[1,1,6],[1,1,1],[x,x],pad='SAME')
    Ic_conv3=Conv_layer(Ic_conv2,[5,1,1],[1,1,1],[x,x],pad='SAME')
    Ic_conv4=Conv_layer(Ic_conv3,[1,5,1],[1,1,1],[x,outshapes[1]],pad='SAME')    
    #forth bit
    Id_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],outshapes[0]],pad='SAME')
    
    I_concat=tf.concat(axis=4, values=[Ia_conv1,Ib_conv5,Ic_conv4,Id_conv1])
    return I_concat

def Elephant_foot(intensor):
    shape = intensor.get_shape().as_list()
    #first bit
    Ia_pool1=tf.nn.max_pool3d(intensor, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME') 
    Ia_conv1=Conv_layer(Ia_pool1,[1,1,1],[1,1,1],[shape[4],192],pad='SAME')
    #second bit
    Ib_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],448],pad='SAME')
    Ib_conv2=Conv_layer(Ib_conv1,[2,2,2],[1,1,1],[448,384],pad='SAME')
    Ib1_conv1=Conv_layer(Ib_conv2,[1,1,3],[1,1,1],[384,256],pad='SAME')
    Ib2_conv1=Conv_layer(Ib_conv2,[2,1,1],[1,1,1],[384,256],pad='SAME')
    Ib3_conv1=Conv_layer(Ib_conv2,[1,2,1],[1,1,1],[384,256],pad='SAME')
    #third bit
    Ic_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],384],pad='SAME')
    Ic1_conv1=Conv_layer(Ic_conv1,[1,1,3],[1,1,1],[384,256],pad='SAME')
    Ic2_conv1=Conv_layer(Ic_conv1,[2,1,1],[1,1,1],[384,256],pad='SAME')
    Ic3_conv1=Conv_layer(Ic_conv1,[1,2,1],[1,1,1],[384,256],pad='SAME')
    #forth bit
    Id_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],320],pad='SAME') 
    
    I_concat=tf.concat(axis=4, values=[Ia_conv1,Ib1_conv1,Ib2_conv1,Ib3_conv1,Ic1_conv1,Ic2_conv1,Ic3_conv1,Id_conv1])
    return I_concat
    
#3d normalization routine, specify if we're training or testing model!!!!!
training = tf.placeholder_with_default(True, shape=())

def model(lidars,Ys,step, batchS, keep_prob, record_low):
    xs=tf.placeholder(tf.float32, [None,40,40,105], name='Xinput') 
    ys=tf.placeholder(tf.float32, [None,1], name='Yinput') 
    #effect=tf.placeholder(tf.float32, [None,4], name='effect') 
    
    #Define keep probability for dropout layers
    #the minus 1 has to do with the number of images, the 1 has to do with 'bands'
    x_image=tf.reshape(xs, [-1,40,40,105,1])
    #DEFINE MODEL   
    #with tf.variable_scope("model", reuse=True): 
        ################initial conv layers######################    
    W_conv1=weight_variable([2,2,3,1,32])#filter size 5x5, 1 band, 32 convolutions out
    b_conv1=bias_variable([32])#biases for each of the 32 convolutions    
    conv1= tf.nn.conv3d(x_image, W_conv1, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
    a_conv1 = tf.nn.bias_add(conv1, b_conv1)    
    h_norm1=tf.contrib.layers.batch_norm(a_conv1, data_format='NHWC', center=True,scale=True, is_training=training)
    r_conv1=tf.nn.relu(h_norm1)

    W_conv2=weight_variable([2,2,3,32,32])#filter size 5x5, 1 band, 32 convolutions out
    b_conv2=bias_variable([32])#biases for each of the 32 convolutions    
    conv2= tf.nn.conv3d(r_conv1, W_conv2, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
    a_conv2 = tf.nn.bias_add(conv2, b_conv2)    
    h_norm2=tf.contrib.layers.batch_norm(a_conv2, data_format='NHWC', center=True,scale=True, is_training=training)
    r_conv2=tf.nn.relu(h_norm2)

    W_conv3=weight_variable([2,2,3,32,64])#filter size 5x5, 1 band, 32 convolutions out
    b_conv3=bias_variable([64])#biases for each of the 32 convolutions    
    conv3= tf.nn.conv3d(r_conv2, W_conv3, strides=[1,1,1,1,1], padding='SAME')#convolutions, stride equals 1, 
    a_conv3 = tf.nn.bias_add(conv3, b_conv3)
    h_norm3=tf.contrib.layers.batch_norm(a_conv3, data_format='NHWC', center=True,scale=True, is_training=training)
    r_conv3=tf.nn.relu(h_norm3)

    h_pool3=tf.nn.max_pool3d(r_conv3, ksize=[1,1,1,2,1], strides=[1,1,1,1,1], padding='SAME')

    W_conv4=weight_variable([1,1,1,64,80])#filter size 5x5, 1 band, 32 convolutions out
    b_conv4=bias_variable([80])#biases for each of the 32 convolutions    
    conv4= tf.nn.conv3d(h_pool3, W_conv4, strides=[1,1,1,1,1], padding='SAME')#convolutions, stride equals 1, 
    a_conv4 = tf.nn.bias_add(conv4, b_conv4)    
    h_norm4=tf.contrib.layers.batch_norm(a_conv4, data_format='NHWC', center=True,scale=True, is_training=training)
    r_conv4=tf.nn.relu(h_norm4)

    W_conv5=weight_variable([2,2,3,80,192])#filter size 5x5, 1 band, 32 convolutions out
    b_conv5=bias_variable([192])#biases for each of the 32 convolutions    
    conv5= tf.nn.conv3d(r_conv4, W_conv5, strides=[1,1,1,1,1], padding='SAME')#convolutions, stride equals 1, 
    a_conv5 = tf.nn.bias_add(conv5, b_conv5)    
    h_norm5=tf.contrib.layers.batch_norm(a_conv5, data_format='NHWC', center=True,scale=True, is_training=training)
    r_conv5=tf.nn.relu(h_norm5)

    h_pool5=tf.nn.max_pool3d(r_conv5, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

    In1=Inception_layer1(h_pool5, [64,64,96,32])
    In2=Inception_layer1(In1, [64,64,96,64])
    In3=Inception_layer1(In2, [64,64,96,64])
    
    #Dimensionality Redux
    #first bit
    In4a_pool1=tf.nn.max_pool3d(In3, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    #second bit
    In4b_conv1=Conv_layer(In3,[1,1,1],[1,1,1],[288,64],pad='SAME')
    In4b_conv2=Conv_layer(In4b_conv1,[2,2,2],[1,1,1],[64,96],pad='SAME')
    In4b_conv3=Conv_layer(In4b_conv2,[2,2,2],[2,2,2],[96,96],pad='VALID')
    #third bit
    In4c_conv1=Conv_layer(In3,[2,2,2],[2,2,2],[288,384],pad='VALID')
    #concat
    In4_concat=tf.concat(axis=4, values=[In4a_pool1,In4b_conv3,In4c_conv1])
    
    In5=Inception_layer2(In4_concat, [192,192,192,192], 128)
    In6=Inception_layer2(In5, [192,192,192,192], 160)
    In7=Inception_layer2(In6, [192,192,192,192], 160)
    In8=Inception_layer2(In7, [192,192,192,192], 192)
    
    #Dimensionality Redux
    #first bit
    In9a_pool1=tf.nn.max_pool3d(In8, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    #second bit
    In9b_conv1=Conv_layer(In8,[1,1,1],[1,1,1],[768,192],pad='SAME')
    In9b_conv2=Conv_layer(In9b_conv1,[1,1,6],[1,1,1],[192,192],pad='SAME')
    In9b_conv3=Conv_layer(In9b_conv2,[1,5,1],[1,1,1],[192,192],pad='SAME')
    In9b_conv4=Conv_layer(In9b_conv3,[5,1,1],[1,1,1],[192,192],pad='SAME')
    In9b_conv5=Conv_layer(In9b_conv4,[2,2,2],[2,2,2],[192,192],pad='SAME')
    #third bit
    In9c_conv1=Conv_layer(In8,[1,1,1],[1,1,1],[768,192],pad='SAME')
    In9c_conv2=Conv_layer(In9c_conv1,[2,2,2],[2,2,2],[192,320],pad='SAME')
    #concat
    In9_concat=tf.concat(axis=4, values=[In9a_pool1,In9b_conv5,In9c_conv2])    
    
    #Dimensionality Redux
    #first bit
    In10a_pool1=tf.nn.max_pool3d(In9_concat, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    #second bit
    In10b_conv1=Conv_layer(In9_concat,[1,1,1],[1,1,1],[1280,192],pad='SAME')
    In10b_conv2=Conv_layer(In10b_conv1,[1,1,6],[1,1,1],[192,192],pad='SAME')
    In10b_conv3=Conv_layer(In10b_conv2,[1,5,1],[1,1,1],[192,192],pad='SAME')
    In10b_conv4=Conv_layer(In10b_conv3,[5,1,1],[1,1,1],[192,192],pad='SAME')
    In10b_conv5=Conv_layer(In10b_conv4,[2,2,2],[2,2,2],[192,192],pad='SAME')
    #third bit
    In10c_conv1=Conv_layer(In9_concat,[1,1,1],[1,1,1],[1280,192],pad='SAME')
    In10c_conv2=Conv_layer(In10c_conv1,[2,2,2],[2,2,2],[192,320],pad='SAME')
    #concat
    In10_concat=tf.concat(axis=4, values=[In10a_pool1,In10b_conv5,In10c_conv2])   
    
    In11=Elephant_foot(In10_concat)
    In12=Elephant_foot(In11)
    #In12=Elephant_foot(In11)
    
    FC_pool=tf.nn.avg_pool3d(In12, ksize=[1,3,3,6,1], strides=[1,1,1,1,1], padding='VALID')
    shape = FC_pool.get_shape().as_list()
    FC_flat=tf.reshape(FC_pool, [-1, shape[1] * shape[2] * shape[3]* shape[4]])

    W_FCo=weight_variable([4096, 1])
    b_FCo=bias_variable([1])
    prediction=(tf.matmul(FC_flat,W_FCo) + b_FCo)#output

    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    train_step=tf.train.AdamOptimizer(step).minimize(loss)

    ###########################FEED IN DATA, ECT..####################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    #sess = tf.Session()#config=config)
    init=tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3)
    sess.run(init)
    
    indices=np.load('H:/Temp/convnet_3d/Leaf_On_Data/withheld.npy')
    withheld_xs=lidars[indices]
    #test_xs=withheld_xs[0:1000]
    validation_xs=withheld_xs[1000::]
    
    withheld_ys=Ys[indices]
    #test_ys=withheld_ys[0:1000]
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
    
    fd = open('H:/Temp/convnet_3d/InceptionV3/summarized.csv','a')

    overtrain_indicator=0
    for i in range(1500000):
        indices=np.random.randint(low=0, high=len(train_ys), size=[batchS,])
        batch_xs=train_xs[indices]
        batch_ys=train_ys[indices]
        #batch_es=train_Es[indices]
        #batch_xs, batch_ys=mnist.train.next_batch(1000)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i%100==0:
            #record training accuracy
            train_acc= sess.run(
                    loss, 
                    feed_dict={xs: batch_xs, ys: batch_ys})
            #summary_writer.add_summary(train_summ, i) 
            #record validation accuracy
            
            valid_accuracies=[]
            for n in range(40):
                batch_valid_xs=validation_xs[25*n:25*(n+1)]
                batch_valid_ys=validation_ys[25*n:25*(n+1)]
                #batch_valid_es=validation_effects[25*n:25*(n+1)]
                
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
            if i>1000:
                if valid_acc2 < record_low:
                    record_low=valid_acc2
                    saver.save(sess, 'H:/Temp/convnet_3d/InceptionV3/InceptionV3_'+str(train_acc)+'_'+str(valid_acc2)+'_'+str(batchS), global_step=i)
                if float(valid_acc2)/train_acc > 1.7:
                    overtrain_indicator+=1
                else:
                    overtrain_indicator=0
                if overtrain_indicator == 5000:
                        break
            print(i, train_acc, valid_acc2, overtrain_indicator) 
    sess.close()
    print ('Done First! Record low: '+str(record_low)) 
    fd.close()
    return record_low

model(lidars,Ys,step=.000005, batchS=15, keep_prob=.5, record_low=320000)