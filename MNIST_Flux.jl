using Flux
using Flux: train!, throttle, @epochs, onecold
using Flux.Losses: crossentropy
using Flux.Data: DataLoader
using Flux.Data.MNIST
using Statistics
using BSON: @save

#Train data
images_train = MNIST.images(:train)
labels_train = MNIST.labels(:train)

#Test data
images_test = MNIST.images(:test)
labels_test = MNIST.labels(:test)

preprocess(img) = Float32.(img)[:]

xtrain = hcat(preprocess.(images_train)...)
xtest = hcat(preprocess.(images_test)...)

#ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
ytrain = hcat([ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels_train ]...)
ytest = hcat([ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels_test ]...)

train = DataLoader((xtrain, ytrain), batchsize = 3000, shuffle = true)

model = Chain(
    Dense(784, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10), softmax
)

loss(x, y) = crossentropy(model(x), y)

ps = params(model)

opt = ADAM()

train!(loss, ps, train, opt)

function upd_loss()
    loss_train = loss(xtrain, ytrain)
    loss_test = loss(xtest, ytest)
    println("Train loss: $(round(loss_train, digits = 6)) | Test loss: $(round(loss_test, digits = 6))")
end
throtle_cb = throttle(upd_loss, 1)
@epochs 100 train!(loss, ps, train, opt, cb = throtle_cb)

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y)) 
accuracy(xtrain, ytrain)
accuracy(xtest, ytest)

function prediction(x)
    pred_vector = model(x) 
    pred_val = findfirst(x -> x == maximum(pred_vector), pred_vector) - 1
    return pred_val
end

prediction(xtest[:, 3])

labels_test[3]

@save "MNIST_model.bson" model