# Representation Learning for Sketches using Combined Representations

## About
In this project, a representation space is learned for sketch classification and providing suggestions of similar sketches on a subspace of the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). This project utilizes both temporal and image form of sketches in a combined model.

Sample sketches of the dataset:
![Quick, Draw!](/images/quickdraw.png?raw=true)

For this project a subspace consisting of 10 classes are used:
* 5 fruit classes:   apple, pear, blackberry, grapes, banana
* 5 vehicle classes: bus, van, truck, airplane, helicopter

For instructions on how to use this code, please see this [README](/src/README.md).

## Models 
Different models using both temporal and image data of sketches are used:

![Representations](/images/representations.png?raw=true).

A final model uses both input representations(image+temporal) to create an embedding/representation for image classification and making suggestions:

![Model](/images/combined.png)

## Image Models: Triplet Networks
Triplet networks in combination with the convolutional networks VGG-16, Inceptionv4 and ResNet-v2 are used to generate a representation space.

### Loss function
A softmax based version is used to calculate the triplet loss function

![Softmax-loss](/images/softmax-loss.png?raw=true)

### Results
ResNet-v2 performs better in matching samples of the same class closer together:

![Triplet-accuracy](/images/triplet_accuracy.png?raw=true)

But fails to generalize well as seen in classifcation accuracy and tSNE embedding:

![Triplet-classification1](images/triplet_classification1.png)

![tSNE1](images/tsne1.png)

### Modifications
To achieve better generalization the following modifications were made

#### Sampling: Semihard Triplets
Instead of sampling triplets randomly, only semi-hard triplets are chosen:

![Triplet-semihard](images/semihard.png)

#### Loss function: Hinge-Loss
![Triplet-hingeloss](images/hinge-loss.png)

### Results after Modification
Classification accuracy: 

![Triplet-classification2](/images/triplet_classification2.png)

t-SNE visualization:
![Triplet-tSNE2](/images/tsne2.png)

### Triplet Network vs Direct Training
![Triplet-vs-Direct](/images/triplet_vs_direct.png)


## Results 
As a temporal model an LSTM with two stacked cells are used. The convolutional model is either trained end-to-end (together with the temporal) model or transferred after training the triplet network.

### Classification
![Classification](images/classification.png)
### Suggestions
Neighbors are of the same class, but more often stilistically different(red) than of the same style(green)s: 
![Nearest-Neighbors](images/neighbors.png)
### KNN classification
![KNN-Classification](images/knn.png)
