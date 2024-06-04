abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

function reset_gradients!(graph)
    for node in graph
        node.gradient = nothing
    end
end


compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end


update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

import Base: ^ , *, +, -, /

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
forward(::ScalarOperator{typeof(+)}, x, y) = x + y
backward(::ScalarOperator{typeof(+)}, x, y, gradient) = (gradient, gradient)

-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
forward(::ScalarOperator{typeof(-)}, x, y) = x - y
backward(::ScalarOperator{typeof(-)}, x, y, gradient) = (gradient, -gradient)

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
forward(::ScalarOperator{typeof(*)}, x, y) = x * y
backward(::ScalarOperator{typeof(*)}, x, y, gradient) = (y' * gradient, x' * gradient)

/(x::GraphNode, y::GraphNode) = ScalarOperator(/, x, y)
forward(::ScalarOperator{typeof(/)}, x, y) = x / y
backward(::ScalarOperator{typeof(/)}, x, y, gradient) = (gradient / y, gradient / y)


import Base: sin , max, min, log
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

log(x::GraphNode) = ScalarOperator(log, x)
forward(::ScalarOperator{typeof(log)}, x) = log(x)
backward(::ScalarOperator{typeof(log)}, x, gradient) = (gradient / x)

max(x::GraphNode, y::GraphNode) = ScalarOperator(max, x, y)
forward(::ScalarOperator{typeof(max)}, x, y) = max(x, y)
backward(::ScalarOperator{typeof(max)}, x, y, gradient) = (gradient * isless(y, x), gradient * isless(x, y))

min(x::GraphNode, y::GraphNode) = ScalarOperator(min, x, y)
forward(::ScalarOperator{typeof(min)}, x, y) = min(x, y)
backward(::ScalarOperator{typeof(min)}, x, y, gradient) = (gradient * isless(x, y), gradient * isless(y, x))

relu(x::GraphNode) = ScalarOperator(relu, x)
forward(::ScalarOperator{typeof(relu)}, x) = max(x, 0)
backward(::ScalarOperator{typeof(relu)}, x, gradient) = gradient * isless(0, x)

logistic(x::GraphNode) = ScalarOperator(logistic, x)
forward(::ScalarOperator{typeof(logistic)}, x) = 1 / (1 + exp(-x))
backward(::ScalarOperator{typeof(logistic)}, x, gradient) = gradient * exp(-x) / (1 + exp(-x))^2



import Base: *
import LinearAlgebra: mul!, diagm
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

import Base: tanh
Base.Broadcast.broadcasted(tanh, x::GraphNode)= BroadcastedOperator(tanh,x; name="tanh")
forward(::BroadcastedOperator{typeof(tanh)},x) = return tanh.(x)
backward(::BroadcastedOperator{typeof(tanh)},x, g) = let 
    𝟙 = ones(length(node.output))
    tuple((𝟙 - tanh.(x).^2) * g)
end


RNN_cell(wx::GraphNode, wh::GraphNode, b::GraphNode, x, h::GraphNode) = BroadcastedOperator(RNN_cell, wx, wh, b, x, h; name="RNN_cell")
forward(::BroadcastedOperator{typeof(RNN_cell)}, wx, wh, b, x ,h) = let 
    tmp = wx * x .+ wh * h .+ b
    return tanh.(tmp)
end
backward(::BroadcastedOperator{typeof(RNN_cell)}, wx, wh, b, x, h, g) = let 
    tmp = wx * x .+ wh * h .+ b
    dtanh = 1 .- tanh.(tmp).^2
    g = g .* dtanh
    tuple(g * x', g * h', sum(g,dims=2), wx' *g, wh' * g)
end

dense(w::GraphNode, b::GraphNode, x::GraphNode) = BroadcastedOperator(dense, w , b, x; name="dense")
forward(::BroadcastedOperator{typeof(dense)}, w, b , x) = let 
    w * x .+ b
end
backward(::BroadcastedOperator{typeof(dense)}, w, b, x, g) = let 
    tuple(g * x', sum(g,dims=2), w' * g)
end

cross_entropy_loss(yhat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, yhat, y; name="cross_entropy_loss")
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, ŷ, y) =
let
    eps = 1e-8
    ŷ = ŷ .- maximum(ŷ; dims=1)
    softmax = exp.(ŷ) ./ sum(exp.(ŷ); dims=1)
    softmax = clamp.(softmax, eps, 1.0 - eps)
    loss = -sum(y .* log.(softmax .+eps); dims=1) 
    # print(mean(loss))
    return mean(loss)
end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, yhat, y, g) =
let
    eps = 1e-8
    yhat = yhat .- maximum(yhat; dims=1)  # for numerical stability
    softmax = exp.(yhat) ./ sum(exp.(yhat); dims=1)
    softmax = clamp.(softmax, eps, 1.0 - eps)
    grad_yhat = softmax - y
    return (g .* grad_yhat,)
end

using MLDatasets
using Statistics: mean  # standard library
train_data = MLDatasets.MNIST(split=:train)
test_data = MLDatasets.MNIST(split=:test)


using Random
function loader(data; batchsize::Int=1, shuffle::Bool=true)
    # Reshape the features
    x1dim = reshape(data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
    
    # One-hot encode the targets
    yhot = onehotbatch(data.targets, 0:9) # make a 10×60000 one-hot matrix
    
    # Combine features and targets
    dataset = (x1dim, yhot)
    
    # Number of samples
    num_samples = size(x1dim, 2)
    
    # Create batches
    function create_batches()
        indices = shuffle ? Random.shuffle(1:num_samples) : 1:num_samples
        batches = []
        
        for i in 1:batchsize:num_samples
            end_idx = min(i+batchsize-1, num_samples)
            push!(batches, (x1dim[:, indices[i:end_idx]], yhot[:, indices[i:end_idx]]))
        end
        
        return batches
    end
    
    return create_batches()
end


function onehotbatch(targets, classes)
    onehot = zeros(Int, length(classes), length(targets))
    for (i, target) in enumerate(targets)
        onehot[target+1, i] = 1
    end
    return onehot
end

function onecold(y)
    return argmax(y, dims=1)
end

function Recurent_stage(Wx::GraphNode, Wh::GraphNode, b1::GraphNode, x, h)
    x̂ = RNN_cell(Wx, Wh, b1, x, h)
    x̂.name = "x̂"
    x̂
end

function dense(w, b, x)
    w * x .+ b
end

function net_test(data, Wx, Wh, W, b1, b2)
    (x,y) = loader(data; batchsize=length(data)) |> first
    x_var = Variable(x[1:196,:], name="x")
    y = Variable(y, name="y")
    h = Recurent_stage(Wx, Wh, b1,  x_var, Variable(zeros(64,length(data))))
    x_var = Variable(x[197:392,:], name="x")
    h =  Recurent_stage(Wx, Wh,  b1,  x_var, h)
    x_var = Variable(x[393:588,:], name="x")
    h =  Recurent_stage(Wx, Wh,  b1,  x_var, h)
    x_var = Variable(x[589:end,:], name="x")
    x̂ =  Recurent_stage(Wx, Wh,  b1, x_var, h)
    ŷ = dense(W, b2, x̂)
    E = cross_entropy_loss(ŷ, y)
    E.name = "loss"
    E.inputs
    graph = topological_sort(E)
    loss = forward!(graph)
    acc = round(100*mean(onecold(ŷ.output) .== onecold(y.output)); digits=2)
    (; loss, acc, split=data.split)
end

function weights_update(graph::Vector, lr=0.0001)
    for node in graph
        if isa(node, Variable) && !isnothing(node.gradient)
            node.output .-= lr*node.gradient
            node.gradient .= 0.0
        end
    end
end

function cliping(graph::Vector)
    clip_value = 5.0 #5.0 best val
    # After backward pass, before weights update
    for node in graph
        if isa(node, Variable) && !isnothing(node.gradient)
            
            node.gradient .= clamp.(node.gradient, -clip_value, clip_value)
        end
    end
end

# Function to print gradients for debugging
function print_gradients(params)
    for p in params
        if p.gradient !== nothing
            println("Gradient for ", p.name, ": ", mean(abs.(p.gradient)))
        end
    end
end

input_size = 14*14
hidden_size = 64
output_size = 10

function xavier_init(out_size, in_size, gain)
    return randn(out_size, in_size) .* gain * sqrt(6.0 / (in_size + out_size))
end

global Wx = Variable(xavier_init(hidden_size, input_size,2), name="Wx")
global Wh = Variable(xavier_init(hidden_size, hidden_size,2), name="Wh")
global W  = Variable(xavier_init(output_size, hidden_size,2), name="W")
global b1 = Variable(randn(hidden_size), name="b1")
global b2 = Variable(randn(output_size), name="b2")

function train(train_data,test_data, Wx, Wh, W, b1, b2)
    best_acc = 0.0
    last_improvent = 0
    lr = 1e-4
    for epoch in 1:30
        for (x,y) in loader(train_data; batchsize = 100)
            h = Variable(zeros(64,100), name="h")
            x_var = Variable(x[1:196,:], name="x")
            y = Variable(y, name="y")
            h = Recurent_stage(Wx, Wh, b1,  x_var, h)
            h.name = "h"
            x_var = Variable(x[197:392,:], name="x")
            h =  Recurent_stage(Wx, Wh,  b1,  x_var, h)
            h.name = "h"
            x_var = Variable(x[393:588,:], name="x")
            h =  Recurent_stage(Wx, Wh,  b1,  x_var, h)
            h.name = "h"
            x_var = Variable(x[589:end,:], name="x")
            x̂ =  Recurent_stage(Wx, Wh,  b1, x_var, h)
            h.name = "x̂"
            ŷ = dense(W, b2, x̂)
            E = cross_entropy_loss(ŷ, y)
            E.name = "loss"
            graph = topological_sort(E)
            forward!(graph)
            reset_gradients!(graph)
            backward!(graph)
            cliping(graph)    
            weights_update(graph, lr)
                
        end
        loss_train, acc_train, _ = net_test(train_data, Wx, Wh, W, b1, b2)
        loss_test, acc_test, _ = net_test(test_data, Wx, Wh, W, b1, b2)

        if acc_train > best_acc
            best_acc = acc_train
            last_improvent = epoch
        end
        if  epoch - last_improvent >= 10 && lr > 1e-6
            lr /= 10
            last_improvent = epoch
        end
        if epoch - last_improvent >= 20
            @warn "Early stopping no inprovement in 20 epochs"
            break
        end
        @info "Epoch: $epoch, Train Loss: $loss_train, Train Accuracy: $acc_train, Test Loss: $loss_test, Test Accuracy: $acc_test"
        # println("Train Loss: $loss, Train Accuracy: $acc")
        # @show loss, acc
    end
end
start_time = time()
@time @info @allocated train(train_data, test_data, Wx, Wh, W, b1, b2)
end_time = time()
elapsed_time = end_time - start_time
@info "Elapsed time: $elapsed_time"
