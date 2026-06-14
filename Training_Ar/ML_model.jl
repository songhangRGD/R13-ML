using Flux
using BSON: @save, @load
using Statistics
using Random
using CUDA

# ================== 保持不变的部分 ================== #
function moving_average_filter(data::AbstractMatrix{Float32}, window_size::Int)
    filtered_data = similar(data)
    rows, cols = size(data)
    half_window = window_size ÷ 2
    for i in 1:rows
        for j in 1:cols
            start_idx = max(1, j - half_window)
            end_idx = min(cols, j + half_window)
            filtered_data[i, j] = mean(data[i, start_idx:end_idx])
        end
    end
    return filtered_data
end

function load_data(base_path::String; window_size=7)
    # 存储每个文件处理后的数据
    filtered_inputs_moments = []
    filtered_outputs_moments = []
    filtered_inputs_source = []
    filtered_outputs_source = []

    # 获取排序后的目录列表
    dirpaths = sort!(readdir(base_path, join=true))
    
    for dirpath in dirpaths
        isdir(dirpath) || continue
        
        # 获取排序后的文件列表
        file_paths = sort!(readdir(dirpath, join=true))
        
        for file_path in file_paths
            if endswith(file_path, "_input.dat")
                # 读取单个文件的数据
                file_inputs_moments = Float32[]
                file_outputs_moments = Float32[]
                file_inputs_source = Float32[]
                file_outputs_source = Float32[]
                
                open(file_path, "r") do file
                    readline(file); readline(file)  # 跳过前两行
                    for line in eachline(file)
                        values = parse.(Float32, split(line))
                        if length(values) == 14
                            # 追加当前文件的数据
                            append!(file_inputs_moments, values[2:10])
                            append!(file_outputs_moments, values[11:12])
                            append!(file_inputs_source, values[2:5])
                            append!(file_outputs_source, values[13:14])
                        end
                    end
                end
                
                # 对单个文件数据独立滤波
                n_points = length(file_inputs_moments) ÷ 9
                if n_points > 0
                    # 重塑为矩阵并滤波
                    input_moments = moving_average_filter(reshape(file_inputs_moments, 9, n_points), window_size)
                    output_moments = moving_average_filter(reshape(file_outputs_moments, 2, n_points), window_size)
                    input_source = moving_average_filter(reshape(file_inputs_source, 4, n_points), window_size)
                    output_source = moving_average_filter(reshape(file_outputs_source, 2, n_points), window_size)
                    
                    # 收集滤波后的数据
                    push!(filtered_inputs_moments, input_moments)
                    push!(filtered_outputs_moments, output_moments)
                    push!(filtered_inputs_source, input_source)
                    push!(filtered_outputs_source, output_source)
                end
            end
        end
    end
    
    # 水平拼接所有文件数据（保留原始形状）
    input_matrix_Ar_1D_moments = hcat(filtered_inputs_moments...)
    output_matrix_Ar_1D_moments = hcat(filtered_outputs_moments...)
    input_matrix_Ar_1D_source = hcat(filtered_inputs_source...)
    output_matrix_Ar_1D_source = hcat(filtered_outputs_source...)
    
    return input_matrix_Ar_1D_moments, output_matrix_Ar_1D_moments, 
           input_matrix_Ar_1D_source, output_matrix_Ar_1D_source
end

function calculate_weights_mv1v1v1(output_data::AbstractMatrix{Float32})
    weights = 1.0 .* (abs.(output_data[1, :])).^0.5 .+ 0.001
    return repeat(weights', 1, 1)
end

function calculate_weights_mvvv1v1(output_data::AbstractMatrix{Float32})
    weights = 1.0 .* (abs.(output_data[1, :])).^0.75 .+ 0.02
    return repeat(weights', 1, 1)
end

function calculate_weights_s4(output_data::AbstractMatrix{Float32})
    weights = 1.0 .* (abs.(output_data[1, :])).^0.5 .+ 0.0005
    return repeat(weights', 1, 1)
end

function calculate_weights_s5(output_data::AbstractMatrix{Float32})
    weights = 1.0 .* (abs.(output_data[1, :])).^1.0 .+ 0.002
    return repeat(weights', 1, 1)
end

function weighted_smooth_l1_loss(pred, target, weights; delta=1.0f0)
    diff = abs.(pred .- target)
    losses = ifelse.(diff .< delta, 0.5f0 .* diff.^2, delta .* diff .- 0.5f0 .* delta^2)
    return sum(weights .* losses) / sum(weights)
end

# ================== 修改后的训练函数 ================== #
function train_mv1v1v1_model(input_data, output_data; epochs=6000, model_file="model_Ar_1D_mv1v1v1.bson")
    input_gpu = gpu(input_data)
    output_gpu = gpu(output_data[1:1, :])  # 选择第一个输出参数

    model = Chain(
        Dense(9, 128, softplus),
        Dense(128, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 1)  # 输出维度改为1
    ) |> gpu

    opt = Flux.setup(Adam(0.0002f0), model)

    open("loss_mv1v1v1.dat", "w") do f
        println(f, "TITLE = \"mv1v1v1 Loss\"")
        println(f, "VARIABLES = \"Epoch\", \"Loss\"")

        for epoch in 1:epochs
            indices = shuffle(1:size(input_gpu, 2))
            X = input_gpu[:, indices]
            Y = output_gpu[:, indices]
            
            # 计算单参数权重
            weights = calculate_weights_mv1v1v1(Y) |> gpu

            loss, grads = Flux.withgradient(model) do m
                weighted_smooth_l1_loss(m(X), Y, weights)
            end

            Flux.update!(opt, model, grads[1])

            if epoch % 20 == 0
                println(f, "$epoch $loss")
                println("Epoch $epoch, Loss_mv1v1v1: $loss")
            end
        end
    end

    model_Ar_1D_mv1v1v1 = cpu(model)
    @save model_file model_Ar_1D_mv1v1v1
    println("mv1v1v1 model saved")
end

function train_mvvv1v1_model(input_data, output_data; epochs=10000, model_file="model_Ar_1D_mvvv1v1.bson")
    input_gpu = gpu(input_data)
    output_gpu = gpu(output_data[2:2, :])  # 选择第二个输出参数

    model = Chain(
        Dense(9, 128, softplus),
        Dense(128, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 1)  # 输出维度改为1
    ) |> gpu

    opt = Flux.setup(Adam(0.0001f0), model)

    open("loss_mvvv1v1.dat", "w") do f
        println(f, "TITLE = \"mvvv1v1 Loss\"")
        println(f, "VARIABLES = \"Epoch\", \"Loss\"")

        for epoch in 1:epochs
            indices = shuffle(1:size(input_gpu, 2))
            X = input_gpu[:, indices]
            Y = output_gpu[:, indices]
            
            weights = calculate_weights_mvvv1v1(Y) |> gpu

            loss, grads = Flux.withgradient(model) do m
                weighted_smooth_l1_loss(m(X), Y, weights)
            end

            Flux.update!(opt, model, grads[1])

            if epoch % 20 == 0
                println(f, "$epoch $loss")
                println("Epoch $epoch, Loss_mvvv1v1: $loss")
            end
        end
    end

    model_Ar_1D_mvvv1v1 = cpu(model)
    @save model_file model_Ar_1D_mvvv1v1
    println("mvvv1v1 model saved")
end

function train_s4_model(input_data, output_data; epochs=6000, model_file="model_Ar_1D_s4.bson")
    input_gpu = gpu(input_data)
    output_gpu = gpu(output_data[1:1, :])  # 选择第一个输出参数

    model = Chain(
        Dense(4, 128, softplus),
        Dense(128, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 1)  # 输出维度改为1
    ) |> gpu

    opt = Flux.setup(Adam(0.0002f0), model)

    open("loss_s4.dat", "w") do f
        println(f, "TITLE = \"S4 Loss\"")
        println(f, "VARIABLES = \"Epoch\", \"Loss\"")

        for epoch in 1:epochs
            indices = shuffle(1:size(input_gpu, 2))
            X = input_gpu[:, indices]
            Y = output_gpu[:, indices]
            
            weights = calculate_weights_s4(Y) |> gpu

            loss, grads = Flux.withgradient(model) do m
                weighted_smooth_l1_loss(m(X), Y, weights)
            end

            Flux.update!(opt, model, grads[1])

            if epoch % 20 == 0
                println(f, "$epoch $loss")
                println("Epoch $epoch, Loss_s4: $loss")
            end
        end
    end

    model_Ar_1D_s4 = cpu(model)
    @save model_file model_Ar_1D_s4
    println("s4 model saved")
end

function train_s5_model(input_data, output_data; epochs=6000, model_file="model_Ar_1D_s5.bson")
    input_gpu = gpu(input_data)
    output_gpu = gpu(output_data[2:2, :])  # 选择第二个输出参数

    model = Chain(
        Dense(4, 128, softplus),
        Dense(128, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 64, softplus),
        Dense(64, 1)  # 输出维度改为1
    ) |> gpu

    opt = Flux.setup(Adam(0.0002f0), model)

    open("loss_s5.dat", "w") do f
        println(f, "TITLE = \"S5 Loss\"")
        println(f, "VARIABLES = \"Epoch\", \"Loss\"")

        for epoch in 1:epochs
            indices = shuffle(1:size(input_gpu, 2))
            X = input_gpu[:, indices]
            Y = output_gpu[:, indices]
            
            weights = calculate_weights_s5(Y) |> gpu

            loss, grads = Flux.withgradient(model) do m
                weighted_smooth_l1_loss(m(X), Y, weights)
            end

            Flux.update!(opt, model, grads[1])

            if epoch % 20 == 0
                println(f, "$epoch $loss")
                println("Epoch $epoch, Loss_s5: $loss")
            end
        end
    end

    model_Ar_1D_s5 = cpu(model)
    @save model_file model_Ar_1D_s5
    println("s5 model saved")
end

# ================== 主函数 ================== #
function main()
    Random.seed!(1234)
    base_path = "/home/songhang/sparta/examples/Hangshock/Ar"
    
    # 加载数据
    input_moments, output_moments, input_source, output_source = load_data(base_path)

    # 训练四个独立模型
    train_mv1v1v1_model(input_moments, output_moments)
    train_mvvv1v1_model(input_moments, output_moments)
    train_s4_model(input_source, output_source)
    train_s5_model(input_source, output_source)
end

main()