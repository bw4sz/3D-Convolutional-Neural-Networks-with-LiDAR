-Lenet_3d.py is the main script for developing a neural network from point cloud and plot data. It requires feild inventory data and voxelized point clouds with a dimension of (n,40,40,105)

-Saved pre-trained models can be found in the Model_Saves folder

-The Lenet_Model_Reloader.py can be used to restore saved Lenet models and begin retraining them with new data. This is the best option for developing a forest inventory in a new area.

-Lenet is a simple model which is useful for experimenting or validation. InceptionV3 outperformed Lenet and so might be a better option for an operation forest inventory.

