#################################
#Code and saved 3D neural networks used in:
#
#Ayrey E. and Hayes D. (2018) "The Use of Three-Dimensional Convolutional Neural Networks to Interpret LiDAR for Forest Inventory" Remote Sensing
#
#################################
The models and scripts provided here are intended to help users develop a LiDAR forest inventory using convolutional neural networks rather than height metrics and linear modelling. These neural networks have been adapted from 2D versions which have been tremendously successful at interpreting images of cats (and also helping self-driving cars navigate). They perform considerably better than traditional LiDAR-inventory approaches provided the dataset is large and robust.

	-The point_cloud_voxelizer.py script is a good place to start. Here normalized point clouds are read in and voxelized for analysis. The output of this script is a numpy array with values corresponding to the number of points within each voxel. This script is necessary to run any 3D convolutional neural network since voxels are needed to run filters over the data.

	-The InceptionV3-3D_Neural_Network contains code needed to train a 3D Inception-based model for predicting forest attributes from LiDAR, or retrain existing models. This was the best performing model in the paper. Model saves are also provided for warm-starting.

	-The Lenet-3D_Neural_Network contains code to train a 3D LeNet-based model.

	-Other-3D_Neural_Network_Implementations contain the code for Alexnet, Resnet, and Googlenet models. 

	-Parametric_Modelling contains data and code to train traditional LiDAR forest inventory models using height metrics extracted from the USFS's FUSION and both Random Forest and Linear Mixed Models. Results and additional material are also in here.

Please contact me with any questions at elias.ayrey@maine.edu. I'm happy to help with simple problems or consult on the creation of new models. I also have better performing models than these that I might be able to share (not published yet).

Sorry the code is poorly formatted and a bit dense. I'm just a grad student trying to get the job done ASAP!
