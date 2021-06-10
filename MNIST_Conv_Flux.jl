using Flux
using Flux: train!, throttle, @epochs, onecold
using Flux: batch, unsqueeze, onehotbatch
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Flux.Data.MNIST
using Statistics
using BSON: @save

#Carregando os dados de treino
images_train = MNIST.images(:train)
labels_train = MNIST.labels(:train)

#Carregando os dados de teste
images_test = MNIST.images(:test)
labels_test = MNIST.labels(:test)

# Função para converter o vetor de imagens em um vetor de matrizes
preprocess(img) = Float32.(img)

# Preparando os tensors
xtrain_tensor = batch(preprocess.(images_train))
xtest_tensor = batch(preprocess.(images_test))

# Adicionando a camada de channel
xtrain = unsqueeze(xtrain_tensor, 3)
xtest = unsqueeze(xtest_tensor, 3)

# Ou em apenas um comando:
# xtrain = cat(preprocess.(images_train)..., dims = 4)
# xtest = cat(preprocess.(images_test)..., dims = 4)

# Convertendo o vetor de labels em um vetor onehot
ytrain = onehotbatch(labels_train, 0:9)
ytest = onehotbatch(labels_test, 0:9)

# Declarando o modelo
model = Chain(
    Conv((5, 5), 1 => 32, relu, pad = SamePad()),
    Conv((5, 5), 32 => 32, relu, pad = SamePad()),
    MaxPool((2, 2)),
    Dropout(0.25),

    Conv((3, 3), 32 => 64, relu, pad = SamePad()),
    Conv((3, 3), 64 => 64, relu, pad = SamePad()),
    MaxPool((2, 2), stride = (2, 2)),
    Dropout(0.25),

    flatten,
    Dense(256, 10, relu),
    Dropout(0.5),
    Dense(10, 10), softmax
)

# Função de perda para o treinamento do modelo
loss(x, y) = logitcrossentropy(model(x), y)
# Prâmetros do modelo
ps = params(model)
# Carregando os dados de treino
train = DataLoader((xtrain, ytrain), batchsize = 3000, shuffle = true)
# Escolhendo o otimizador
opt = RMSProp()

# Primiero treinamento (mais demorado)
train!(loss, ps, train, opt)

# Função para exibir o andamento do treinamento
function upd_loss()
    loss_train = loss(xtrain, ytrain)
    loss_test = loss(xtest, ytest)
    println("Train loss: $(round(loss_train, digits = 6)) | Test loss: $(round(loss_test, digits = 6))")
end
throtle_cb = throttle(upd_loss, 1) # Função que exibe o resultado de upd_loss() no REPL a cada segundo (1s)

# Treinando o modelo com 30 épocas (qnt de treinos)
@epochs 30 train!(loss, ps, train, opt, cb = throtle_cb)

# Função para medir a acurácia do modelo
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y)) 
accuracy(xtrain, ytrain)
accuracy(xtest, ytest)

# Macro para salvar o modelo já treinado no formato BSON
@save "MNIST_Conv_model.bson" model