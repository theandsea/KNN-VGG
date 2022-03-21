# K-Nearest Neighbor algorithm + VGG
kNN is a popular unsupervised method for classification. Different representation spaces have hugh impact on the accuracy of our kNN classifier. We try to use the last layer of a pretrained deep neural network model to feed the KNN classifier. As comparison, pixels and HoG features was also taken.
The comparison was taken on CIFAR-10.
The model was stored in "cifar10_models/state_dicts/" and takes lots of space. Download and unzip it before run the program.
