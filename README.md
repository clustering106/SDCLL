# LDCEC1
LEARNING LOSS WITH CLUSTER
As one of the current mainstream unsupervised
algorithms, deep clustering is widely used in various fields due
to its excellent clustering performance. However, most of the
deep clustering methods use batch gradient descent optimization
algorithms, which greatly reduces the learning speed of the neural
network, and the loss function is easy to fall into the local
optimal solution, which damages the clustering performance.
Second, the boundary points in unsupervised clustering are
difficult to judge their categories, and unreliable samples near
the cluster boundary may confuse or even mislead the training
process of deep neural networks. To address these issues, we
propose a novel deep clustering method, LDCEC, which attaches
a small parameter module named ”loss prediction module” to a
deep convolutional autoencoder network. The samples with high
confidence are selected by using the features of the hidden layer
of the network through the loss prediction module. Then, in
order to avoid the insufficiency of the batch gradient descent
optimization algorithm, we use the self-paced learning method
to add samples, and select the most confident samples to be
added to the model training in the iterative process, and gradually
increase the number of samples entering the network training
from easy to difficult. Specifically, we use the ”loss prediction
module” to predict the loss of the input data, thus adding its
selected relatively confident samples (with a small loss) to the
model for training, and gradually select more samples after
many iterations Enter network training. We demonstrate that our
algorithm outperforms state-of-the-art work through experiments
on four image datasets.
