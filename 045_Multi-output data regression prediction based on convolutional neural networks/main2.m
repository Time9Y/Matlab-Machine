%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');

%%  划分训练集和测试集
temp = randperm(719);

P_train = res(temp(1: 500), 1 : 28)';
T_train = res(temp(1: 500), 29: 31)';
M = size(P_train, 2);

P_test = res(temp(501: end), 1 : 28)';
T_test = res(temp(501: end), 29: 31)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
p_train =  double(reshape(p_train, 7, 4, 1, M));
p_test  =  double(reshape(p_test , 7, 4, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%%  构造网络结构
layers = [
 imageInputLayer([7, 4, 1])             % 输入层输入数据为7 * 4
 
 convolution2dLayer([2, 1], 8)          % 第一个卷积层 卷积核大小为2 * 1
 batchNormalizationLayer                % 批归一化层
 reluLayer                              % relu层 激活函数成
                                    
 convolution2dLayer([2, 2], 16)         % 第二个卷积层 卷积核大小为 2 * 2
 batchNormalizationLayer                % 批归一化层
 reluLayer                              % relu 激活层

 convolution2dLayer([2, 2], 32)         % 第二个卷积层 卷积核大小为 2 * 2
 batchNormalizationLayer                % 批归一化层
 reluLayer                              % relu 激活层

 fullyConnectedLayer(128)               % 全连接层 神经元个数为128个
 reluLayer                              % relu 激活层
 
 fullyConnectedLayer(3)                 % 输出层
 regressionLayer];

%%  参数设置
options = trainingOptions('adam', ...   % 梯度计算方法为Adam
    'MiniBatchSize', 64, ...            % 批训练 每次训练64个样本
    'MaxEpochs', 200, ...               % 最大训练次数为200次
    'InitialLearnRate', 0.001, ...      % 初始学习率为0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...     % 调整后学习率为0.001 * 0.1
    'LearnRateDropPeriod', 150, ...     % 训练150次后 学习率进行调整
    'Shuffle', 'every-epoch', ...       % 每次训练打乱顺序
    'ValidationPatience', Inf, ...      % 关闭验证
    'Plots', 'training-progress', ...   % 画出训练曲线
    'ExecutionEnvironment', 'cpu', ...  % 采用CPU运行
    'Verbose', false);                  % 关闭命令行显示

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  模型预测
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
T_sim2 = mapminmax('reverse', t_sim2', ps_output);

%%  绘制网络分析图
analyzeNetwork(layers)

for i = 1: 3

    %%  均方根误差
    error1(i, :) = sqrt(sum((T_sim1(i, :) - T_train(i, :)).^2) ./ M);
    error2(i, :) = sqrt(sum((T_sim2(i, :) - T_test (i, :)).^2) ./ N);
    
    %%  绘图
    figure
    subplot(2, 1, 1)
    plot(1: M, T_train(i, :), 'r-*', 1: M, T_sim1(i, :), 'b-o', 'LineWidth', 1)
    legend('真实值', '预测值')
    xlabel('预测样本')
    ylabel('预测结果')
    string = {'训练集预测结果对比'; ['RMSE=' num2str(error1(i, :))]};
    title(string)
    xlim([1, M])
    grid
    
    subplot(2, 1, 2)
    plot(1: N, T_test(i, :), 'r-*', 1: N, T_sim2(i, :), 'b-o', 'LineWidth', 1)
    legend('真实值', '预测值')
    xlabel('预测样本')
    ylabel('预测结果')
    string = {'测试集预测结果对比'; ['RMSE=' num2str(error2(i, :))]};
    title(string)
    xlim([1, N])
    grid
    
    %%  分割线
    disp('**************************')
    disp(['下列是输出', num2str(i)])
    disp('**************************')
    
    %%  相关指标计算
    % 决定系数 R2
    R1(i, :) = 1 - norm(T_train(i, :) - T_sim1(i, :))^2 / norm(T_train(i, :) - mean(T_train(i, :)))^2;
    R2(i, :) = 1 - norm(T_test (i, :) - T_sim2(i, :))^2 / norm(T_test (i, :) - mean(T_test (i, :)))^2;
    
    disp(['训练集数据的R2为：', num2str(R1(i, :))])
    disp(['测试集数据的R2为：', num2str(R2(i, :))])
    
    % 平均绝对误差 MAE
    mae1(i, :) = sum(abs(T_sim1(i, :) - T_train(i, :))) ./ M ;
    mae2(i, :) = sum(abs(T_sim2(i, :) - T_test (i, :))) ./ N ;
    
    disp(['训练集数据的MAE为：', num2str(mae1(i, :))])
    disp(['测试集数据的MAE为：', num2str(mae2(i, :))])
    
    % 平均相对误差 MBE
    mbe1(i, :) = sum(T_sim1(i, :) - T_train(i, :)) ./ M ;
    mbe2(i, :) = sum(T_sim2(i, :) - T_test (i, :)) ./ N ;
    
    disp(['训练集数据的MBE为：', num2str(mbe1(i, :))])
    disp(['测试集数据的MBE为：', num2str(mbe2(i, :))])

end