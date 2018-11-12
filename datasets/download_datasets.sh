# Download and extract CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
gunzip cifar-10-binary.tar
tar -xf cifar-10-binary.tar
rm cifar-10-binary.tar
# Download and extract MNIST
mkdir MNIST
cd MNIST
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
# Download and extract Faces
wget http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces.tar.Z
tar -xf faces.tar.Z
rm faces.tar.Z
