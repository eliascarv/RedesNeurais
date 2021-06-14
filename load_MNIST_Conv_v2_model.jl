using Flux
using Flux: onecold
using Flux.Data.MNIST
using BSON: @load

images = MNIST.images(:test)
labels = MNIST.labels(:test)

@load "MNIST_Conv_v2_model.bson" model

function prediction(img)
    input = cat(Float32.(img), dims = 4)
    pred_vector = model(input)[:] 
    pred_val = onecold(pred_vector) - 1
    return pred_val
end

images[200]
println("O número da imagem é $(labels[200]).")

println("A rede neural prediz que é o número $(prediction(images[200])).")

