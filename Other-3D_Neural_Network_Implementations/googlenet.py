# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:41:34 2017

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
lidars=lidars.astype('float32')
Ys=Ys.astype('float32')
Ys=Ys.reshape(-1,1)
#Effects=Effects.astype('float32')

xs=tf.placeholder(tf.float32, [None,40,40,105], name='Xinput') 
ys=tf.placeholder(tf.float32, [None,1], name='Yinput') 
#effect=tf.placeholder(tf.float32, [None,4], name='effect') 

#Define keep probability for dropout layers
keep_prob = tf.constant(0.5)
#the minus 1 has to do with the number of images, the 1 has to do with 'bands'
x_image=tf.reshape(xs, [-1,40,40,105,1])

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
    h_conv=tf.nn.elu(added)
    return h_conv

#3d normalization routine, specify if we're training or testing model!!!!!
training = tf.placeholder_with_default(True, shape=())

#DEFINE MODEL   
#with tf.variable_scope("model", reuse=True): 
    ################initial conv layers######################
h_conv1=Conv_layer(x_image,[2,2,3],[2,2,2],[1,64])
h_pool1=tf.nn.max_pool3d(h_conv1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
h_norm1=tf.contrib.layers.batch_norm(h_pool1, data_format='NHWC', center=True,scale=True, is_training=training)

h_conv2=Conv_layer(h_norm1,[1,1,1],[1,1,1],[64,64],pad='SAME')

h_conv3=Conv_layer(h_conv2,[2,2,2],[1,1,1],[64,192],pad='SAME')
h_pool3=tf.nn.max_pool3d(h_conv3, ksize=[1,2,2,2,1], strides=[1,1,1,2,1], padding='VALID')
h_norm3=tf.contrib.layers.batch_norm(h_pool3, data_format='NHWC', center=True,scale=True, is_training=training,scope='cnn3d-batch_norm3')

##################INCEPTION GROUP ONE#####################
IN1_conv1=Conv_layer(h_norm3,[1,1,1],[1,1,1],[192,64],pad='SAME')

IN1_conv2=Conv_layer(h_norm3,[1,1,1],[1,1,1],[192,96],pad='SAME')
IN1_conv2b=Conv_layer(IN1_conv2,[2,2,2],[1,1,1],[96,128],pad='SAME')

IN1_conv3=Conv_layer(h_norm3,[1,1,1],[1,1,1],[192,16],pad='SAME')
IN1_conv3b=Conv_layer(IN1_conv3,[3,3,3],[1,1,1],[16,32],pad='SAME')

IN1_pool4=tf.nn.max_pool3d(h_norm3, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN1_conv4=Conv_layer(IN1_pool4,[1,1,1],[1,1,1],[192,32],pad='SAME')

IN1_concat=tf.concat(axis=4, values=[IN1_conv1,IN1_conv2b,IN1_conv3b,IN1_conv4])

##################INCEPTION GROUP TWO#####################
IN2_conv1=Conv_layer(IN1_concat,[1,1,1],[1,1,1],[256,128],pad='SAME')

IN2_conv2=Conv_layer(IN1_concat,[1,1,1],[1,1,1],[256,128],pad='SAME')
IN2_conv2b=Conv_layer(IN2_conv2,[2,2,2],[1,1,1],[128,192],pad='SAME')

IN2_conv3=Conv_layer(IN1_concat,[1,1,1],[1,1,1],[256,32],pad='SAME')
IN2_conv3b=Conv_layer(IN2_conv3,[3,3,3],[1,1,1],[32,96],pad='SAME')

IN2_pool4=tf.nn.max_pool3d(IN1_concat, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN2_conv4=Conv_layer(IN2_pool4,[1,1,1],[1,1,1],[256,64],pad='SAME')

IN2_concat=tf.concat(axis=4, values=[IN2_conv1,IN2_conv2b,IN2_conv3b,IN2_conv4])
IN2_pool=tf.nn.max_pool3d(IN2_concat, ksize=[1,2,2,2,1], strides=[1,2,2,1,1], padding='VALID')

##################INCEPTION GROUP THREE#####################
IN3_conv1=Conv_layer(IN2_pool,[1,1,1],[1,1,1],[480,192],pad='SAME')

IN3_conv2=Conv_layer(IN2_pool,[1,1,1],[1,1,1],[480,96],pad='SAME')
IN3_conv2b=Conv_layer(IN3_conv2,[2,2,2],[1,1,1],[96,208],pad='SAME')

IN3_conv3=Conv_layer(IN2_pool,[1,1,1],[1,1,1],[480,16],pad='SAME')
IN3_conv3b=Conv_layer(IN3_conv3,[3,3,3],[1,1,1],[16,48],pad='SAME')

IN3_pool4=tf.nn.max_pool3d(IN2_pool, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN3_conv4=Conv_layer(IN3_pool4,[1,1,1],[1,1,1],[480,64],pad='SAME')

IN3_concat=tf.concat(axis=4, values=[IN3_conv1,IN3_conv2b,IN3_conv3b,IN3_conv4])

##################INCEPTION GROUP FOUR#####################
IN4_conv1=Conv_layer(IN3_concat,[1,1,1],[1,1,1],[512,160],pad='SAME')

IN4_conv2=Conv_layer(IN3_concat,[1,1,1],[1,1,1],[512,112],pad='SAME')
IN4_conv2b=Conv_layer(IN4_conv2,[2,2,2],[1,1,1],[112,224],pad='SAME')

IN4_conv3=Conv_layer(IN3_concat,[1,1,1],[1,1,1],[512,24],pad='SAME')
IN4_conv3b=Conv_layer(IN4_conv3,[3,3,3],[1,1,1],[24,64],pad='SAME')

IN4_pool4=tf.nn.max_pool3d(IN3_concat, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN4_conv4=Conv_layer(IN4_pool4,[1,1,1],[1,1,1],[512,64],pad='SAME')

IN4_concat=tf.concat(axis=4, values=[IN4_conv1,IN4_conv2b,IN4_conv3b,IN4_conv4])

##################INCEPTION GROUP FIVE#####################
IN5_conv1=Conv_layer(IN4_concat,[1,1,1],[1,1,1],[512,128],pad='SAME')

IN5_conv2=Conv_layer(IN4_concat,[1,1,1],[1,1,1],[512,128],pad='SAME')
IN5_conv2b=Conv_layer(IN5_conv2,[2,2,2],[1,1,1],[128,256],pad='SAME')

IN5_conv3=Conv_layer(IN4_concat,[1,1,1],[1,1,1],[512,24],pad='SAME')
IN5_conv3b=Conv_layer(IN5_conv3,[3,3,3],[1,1,1],[24,64],pad='SAME')

IN5_pool4=tf.nn.max_pool3d(IN4_concat, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN5_conv4=Conv_layer(IN5_pool4,[1,1,1],[1,1,1],[512,64],pad='SAME')

IN5_concat=tf.concat(axis=4, values=[IN5_conv1,IN5_conv2b,IN5_conv3b,IN5_conv4])

##################INCEPTION GROUP SIX#####################
IN6_conv1=Conv_layer(IN5_concat,[1,1,1],[1,1,1],[512,112],pad='SAME')

IN6_conv2=Conv_layer(IN5_concat,[1,1,1],[1,1,1],[512,144],pad='SAME')
IN6_conv2b=Conv_layer(IN6_conv2,[2,2,2],[1,1,1],[144,288],pad='SAME')

IN6_conv3=Conv_layer(IN5_concat,[1,1,1],[1,1,1],[512,32],pad='SAME')
IN6_conv3b=Conv_layer(IN6_conv3,[3,3,3],[1,1,1],[32,64],pad='SAME')

IN6_pool4=tf.nn.max_pool3d(IN5_concat, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN6_conv4=Conv_layer(IN6_pool4,[1,1,1],[1,1,1],[512,64],pad='SAME')

IN6_concat=tf.concat(axis=4, values=[IN6_conv1,IN6_conv2b,IN6_conv3b,IN6_conv4])

##################INCEPTION GROUP SEVEN#####################
IN7_conv1=Conv_layer(IN6_concat,[1,1,1],[1,1,1],[528,256],pad='SAME')

IN7_conv2=Conv_layer(IN6_concat,[1,1,1],[1,1,1],[528,160],pad='SAME')
IN7_conv2b=Conv_layer(IN7_conv2,[2,2,2],[1,1,1],[160,320],pad='SAME')

IN7_conv3=Conv_layer(IN6_concat,[1,1,1],[1,1,1],[528,32],pad='SAME')
IN7_conv3b=Conv_layer(IN7_conv3,[3,3,3],[1,1,1],[32,128],pad='SAME')

IN7_pool4=tf.nn.max_pool3d(IN6_concat, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN7_conv4=Conv_layer(IN7_pool4,[1,1,1],[1,1,1],[528,128],pad='SAME')

IN7_concat=tf.concat(axis=4, values=[IN7_conv1,IN7_conv2b,IN7_conv3b,IN7_conv4])
IN7_pool=tf.nn.max_pool3d(IN7_concat, ksize=[1,2,2,2,1], strides=[1,1,1,2,1], padding='VALID')

##################INCEPTION GROUP EIGHT#####################
IN8_conv1=Conv_layer(IN7_pool,[1,1,1],[1,1,1],[832,256],pad='SAME')

IN8_conv2=Conv_layer(IN7_pool,[1,1,1],[1,1,1],[832,160],pad='SAME')
IN8_conv2b=Conv_layer(IN8_conv2,[2,2,2],[1,1,1],[160,320],pad='SAME')

IN8_conv3=Conv_layer(IN7_pool,[1,1,1],[1,1,1],[832,32],pad='SAME')
IN8_conv3b=Conv_layer(IN8_conv3,[3,3,3],[1,1,1],[32,128],pad='SAME')

IN8_pool4=tf.nn.max_pool3d(IN7_pool, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN8_conv4=Conv_layer(IN8_pool4,[1,1,1],[1,1,1],[832,128],pad='SAME')

IN8_concat=tf.concat(axis=4, values=[IN8_conv1,IN8_conv2b,IN8_conv3b,IN8_conv4])

##################INCEPTION GROUP NINE#####################
IN9_conv1=Conv_layer(IN8_concat,[1,1,1],[1,1,1],[832,384],pad='SAME')

IN9_conv2=Conv_layer(IN8_concat,[1,1,1],[1,1,1],[832,192],pad='SAME')
IN9_conv2b=Conv_layer(IN9_conv2,[2,2,2],[1,1,1],[192,384],pad='SAME')

IN9_conv3=Conv_layer(IN8_concat,[1,1,1],[1,1,1],[832,48],pad='SAME')
IN9_conv3b=Conv_layer(IN9_conv3,[3,3,3],[1,1,1],[48,128],pad='SAME')

IN9_pool4=tf.nn.max_pool3d(IN8_concat, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
IN9_conv4=Conv_layer(IN9_pool4,[1,1,1],[1,1,1],[832,128],pad='SAME')

IN9_concat=tf.concat(axis=4, values=[IN9_conv1,IN9_conv2b,IN9_conv3b,IN9_conv4])

######################FULLY CONNECTED########################
FC_pool=tf.nn.avg_pool3d(IN9_concat, ksize=[1,3,3,6,1], strides=[1,1,1,1,1], padding='VALID')

shape = FC_pool.get_shape().as_list()
FC_flat=tf.reshape(FC_pool, [-1, shape[1] * shape[2] * shape[3]* shape[4]])
#ADD EFFECTS? NOT RIGHT NOW
#reg_inputs=tf.concat(axis=1, values=[FC_flat,effect])
#shape2 = reg_inputs.get_shape().as_list()

W_FCo=weight_variable([1024, 1])
b_FCo=bias_variable([1])
prediction=(tf.matmul(FC_flat,W_FCo) + b_FCo)#output

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)

###########################FEED IN DATA, ECT..####################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init=tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=3)
sess.run(init)

#indices=np.random.randint(low=0, high=len(Ys), size=[2000,])
indices=np.load('H:/Temp/convnet_3d/Leaf_On_Data/withheld.npy')
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
summary_writer = tf.summary.FileWriter('H:/Temp/convnet_3d/googlenet/Model_saves/Summary')

fd = open('H:/Temp/convnet_3d/googlenet/New Folder/summarized.csv','a')

 
record_low=100000000
overtrain_indicator=0
for i in range(500000):
    indices=np.random.randint(low=0, high=len(train_ys), size=[35,])
    batch_xs=train_xs[indices]
    batch_ys=train_ys[indices]
    #batch_es=train_Es[indices]
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
            #batch_valid_es=validation_effects[25*n:25*(n+1)]
            
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
        if i>1000:
            if valid_acc2 < record_low:
                record_low=valid_acc2
                saver.save(sess, 'H:\Temp\convnet_3d\\googlenet\\Leaf_On\\Googlenet_'+str(train_acc)+'_'+str(valid_acc2), global_step=i)
            if float(valid_acc)/train_acc > 1.5:
                overtrain_indicator+=1
            else:
                overtrain_indicator=0
            if overtrain_indicator == 15:
                    break
        print(i, train_acc, valid_acc2, overtrain_indicator) 
print ('Done! Record low: '+str(record_low)) 

fd.close()