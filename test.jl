using MLDatasets
using Images

train_x, train_y = MNIST.traindata(Float32, 1:10)

image = train_x[:, :, 1]
MNIST.convert2image(image)



