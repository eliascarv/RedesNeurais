using Flux
using Flux: train!, throttle, @epochs, onecold
using Flux: batch, unsqueeze, onehotbatch
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Flux.Data.MNIST
using Statistics, Random
using BSON: @save

Random.seed!(2)

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
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(256, 120, relu), 
    Dense(120, 84, relu), 
    Dense(84, 10)
)

# Função de perda para o treinamento do modelo
loss(x, y) = logitcrossentropy(model(x), y)
# Prâmetros do modelo
ps = params(model)
# Carregando os dados de treino e teste
train = DataLoader((xtrain, ytrain), batchsize = 200, shuffle = true)
test = DataLoader((xtest, ytest), batchsize = 200)
# Escolhendo o otimizador
opt = ADAM()

# Primiero treinamento (mais demorado)
@time train!(loss, ps, train, opt)

# Função para avaliar a loss do modelo
function eval_loss(loader)
    loss_sum = 0.0
    batch_tot = 0
    for batch in loader
        x, y = batch
        loss_sum += loss(x, y) * size(x)[end]
        batch_tot += size(x)[end]
    end
    return round(loss_sum/batch_tot, digits = 4)
end

# Função para exibir a loss do modelo
function evalcb() 
    loss_train = eval_loss(train)
    loss_test = eval_loss(test)
    push!(trainloss_array, loss_train)
    push!(testloss_array, loss_test)
    println("Train loss: $(round(loss_train, digits = 6)) | Test loss: $(round(loss_test, digits = 6))")
end
throttle_cb = throttle(evalcb, 10) # Função que exibe o resultado de upd_loss() no REPL a cada segundo (10s)

# Treinando o modelo com 30 épocas, verificou-se que 15 a 17s épocas seria o ideal. 
const trainloss_array = Float64[]
const testloss_array = Float64[]
@time @epochs 17 train!(loss, ps, train, opt, cb = throttle_cb)

# Função para medir a acurácia do modelo
accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y)) 
# ŷtrin = model(xtrain)
# accuracy(ŷtrain, ytrain)
ŷtest = model(xtest)
accuracy(ŷtest, ytest)

# Macro para salvar o modelo já treinado no formato BSON
# Acurácia de 0.9884 na base de teste
@save "MNIST_Conv_v3_model.bson" model

using Plots
plot(collect(2:length(trainloss_array)), 
     [trainloss_array[2:end] testloss_array[2:end]], 
     labels = ["Train Loss" "Test Loss"],
     lw=2)
