# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:39:20 2017

@author: Elias Ayrey
"""

import tensorflow as tf
import numpy as np
import pandas as pd

#Specify if you want to use the GPU or CPU. The GPU is required for training. For validation or prediction a powerful CPU will do.
num_cores = 40
GPU=False
CPU=True
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0
   
############################ LOAD DATA ######################################   
#Load array of voxelized point cloud grid cells developed using the point_cloud_voxelizer.py
lidars=np.load("H:\\Temp\\convnet_3d\\Leaf_On_Data\\unbuffered_leafon_quarterm.npy").astype("int8")
#Load dependant variable and reformat to a (-1,1) array
Ys=pd.read_csv("H:\\Temp\\convnet_3d\\Leaf_On_Data\\Ys.csv", sep=',')
Ys=Ys['Biomass_AG']
Ys=np.asarray(Ys)
Ys=Ys.astype('float32')
Ys=Ys.reshape(-1,1)

############################ SEGMENT DATA ######################################
indices=np.load('H:/Temp/convnet_3d/Leaf_On_Data/withheld.npy')
withheld_xs=lidars[indices]
withheld_ys=Ys[indices]

#validation data
test_xs=withheld_xs[0:1000]
test_ys=withheld_ys[0:1000]

############################ Initialize Model ######################################
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

sess = tf.Session(config=config)
init=tf.global_variables_initializer()
sess.run(init) 

#Import saved model. You need to import the file and meta file. 
#The checkpoint file that goes along with the model may need to be edited to refer to the correct directory of the model.
saver = tf.train.import_meta_graph("H:\\Temp\\Chapter_2_bits\\Model_Testing\\Model_Saves\\Inception_Experiments\\IRT_Autoencoder_0.027163573_0.20345552_305.0156_0-49900.meta")
saver.restore(sess, "H:\\Temp\\Chapter_2_bits\\Model_Testing\\Model_Saves\\Inception_Experiments\\IRT_Autoencoder_0.027163573_0.20345552_305.0156_0-49900")
graph=tf.get_default_graph()
#just a list of names of tensorflow graph operations. Not necessary but useful for finding an operation
#names=[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
#names=[s for s in names if "BatchNorm"  not in s]
#names=[s for s in names if "truncated_normal" not in s]
#names=[s for s in names if "gradients" not in s]
##names=[s for s in names if "adam" not in s]
##names=[s for s in names if "Adam" not in s]
#names=[s for s in names if "save" not in s]

#extract the model inputs from the graph
xs0=graph.get_tensor_by_name("Xinput:0")
ys0=graph.get_tensor_by_name("Yinput:0")
try:
    xs0=graph.get_tensor_by_name("Xinput_1:0")
    ys0=graph.get_tensor_by_name("Yinput_1:0")
except:
    pass  
training=graph.get_tensor_by_name("PlaceholderWithDefault/input:0")

#extract the model output from the graph
pred1=graph.get_tensor_by_name("add:0")
try:
   pred1=graph.get_tensor_by_name("add_16:0")
except:
   pass

#extract the model loss from the graph (MSE)  
mse=graph.get_tensor_by_name("Mean:0")
train_step=graph.get_operation_by_name("Adam")

#MODEL VALIDATION (Not necessary if you're going to restart training)
########################################################################################## 
#segment the validation data into batches to make it more managable.   
# # valid_accuracies=[]
# # RMSEs=[]
# # for n in range(60):
    # # batch_valid_xs=test_xs[50*n:50*(n+1)]
    # # batch_valid_ys=test_ys[50*n:50*(n+1)]
                    
    # # [valid_acc, pred]= sess.run([mse, pred1] ,feed_dict={xs0: batch_valid_xs, ys0: batch_valid_ys, training: True})
    # # valid_accuracies.append(valid_acc)
                
    # # RMSE= np.sqrt(np.mean((np.asarray(batch_valid_ys).flatten()-np.asarray(pred).flatten())**2))
    # # RMSEs.append(RMSE) 
    # # print(round(n/120*100))
    
# # RMSE=np.mean(RMSEs)     
# # valid_acc2=np.mean(valid_accuracies)    
##########################################################################################

#RESTART MODEL TRAINING (How one might train these models on a new dataset..)
##########################################################################################
#The 'new' data
validation_xs=withheld_xs[1000::]
validation_ys=withheld_ys[1000::]
    
mask = np.ones(len(Ys), np.bool)
mask[indices] = 0
#training dataset
train_xs = lidars[mask]
train_ys = Ys[mask]
    
batchS=25
record_low=10000000000
#summary file
fd = open('H:/Temp/convnet_3d/InceptionV3/summarized.csv','a')
overtrain_indicator=0
for i in range(1500000):
    #for each trainding step, withhold a batch of data, train 
    indices=np.random.randint(low=0, high=len(train_ys), size=[batchS,])
    batch_xs=train_xs[indices]
    batch_ys=train_ys[indices]
    sess.run(train_step, feed_dict={xs0: batch_xs, ys0: batch_ys})
    #run a validation test every 100 steps
    if i%100==0:
        #training accuracy
        train_acc= sess.run(mse, feed_dict={xs0: batch_xs, ys0: batch_ys})
        
        
        #assess model accuracy on validation data in small peices (don't want to overload GPU mem)
        valid_accuracies=[]
        RMSEs=[]
        for n in range(40):
            batch_valid_xs=validation_xs[25*n:25*(n+1)]
            batch_valid_ys=validation_ys[25*n:25*(n+1)]
            
            [valid_acc, pred]= sess.run([mse, pred1] ,feed_dict={xs0: batch_valid_xs, ys0: batch_valid_ys, training: True})
            valid_accuracies.append(valid_acc)
                
            RMSE= np.sqrt(np.mean((np.asarray(batch_valid_ys).flatten()-np.asarray(pred).flatten())**2))
            RMSEs.append(RMSE) 
        RMSE=np.mean(RMSEs)     
        valid_acc2=np.mean(valid_accuracies)    

        if i>1000:
            if valid_acc2 < record_low:
                record_low=valid_acc2
                #save new model
                saver.save(sess, 'H:\\Temp\\Chapter_2_bits\\Model_Testing\\InceptionV3_'+str(train_acc)+'_'+str(valid_acc2)+'_'+str(RMSE), global_step=i)
                overtrain_indicator=0
            else:
                overtrain_indicator+=1
                #stop model training if no progress has been made in past 500 steps (better to do it interactivly)
#                if overtrain_indicator == 5000:
#                        break
        print(i, train_acc, valid_acc2, RMSE, overtrain_indicator) 