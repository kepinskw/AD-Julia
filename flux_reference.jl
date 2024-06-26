using MLDatasets, Flux
train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

function loader(data; batchsize::Int=1)
    x1dim = reshape(data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
    yhot  = Flux.onehotbatch(data.targets, 0:9) # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
end

net = Chain(
    RNN((14 * 14) => 64, tanh),
    Dense(64 => 10, identity),
)

using Statistics: mean  # standard library
function loss_and_accuracy(model, data)
    (x,y) = only(loader(data; batchsize=length(data)))
    Flux.reset!(model)
    ŷ = model(x[  1:196,:])
    ŷ = model(x[197:392,:])
    ŷ = model(x[393:588,:])
    ŷ = model(x[589:end,:])
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  # return a NamedTuple
end

@show loss_and_accuracy(net, test_data);  # accuracy about 10%, before training

train_log = []
settings = (;
    eta = 15e-3,
    epochs = 5,
    batchsize = 100,
)


function train(train_data,settings,net,train_log)
    opt_state = Flux.setup(Descent(settings.eta), net);
    for epoch in 1:settings.epochs
        @time for (x,y) in loader(train_data, batchsize=settings.batchsize)
            Flux.reset!(net)
            grads = Flux.gradient(model -> let
                    ŷ = model(x[  1:196,:])
                    ŷ = model(x[197:392,:])
                    ŷ = model(x[393:588,:])
                    ŷ = model(x[589:end,:])
                    Flux.logitcrossentropy(ŷ, y)
                end, net)
            Flux.update!(opt_state, net, grads[1])
        end
        
        loss, acc, _ = loss_and_accuracy(net, train_data)
        test_loss, test_acc, _ = loss_and_accuracy(net, test_data)
        @info epoch acc test_acc
        nt = (; epoch, loss, acc, test_loss, test_acc) 
        push!(train_log, nt)
    end
    
end
#using ProgressMeter
start_time = time()
@time train(train_data,settings,net,train_log)
end_time = time()
elapsed_time = end_time - start_time
@info "Elapsed time: $elapsed_time"