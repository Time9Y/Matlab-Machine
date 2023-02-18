%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');

%%  划分训练集和测试集
temp = randperm(103);

P_train = res(temp(1: 80), 1: 7)';
T_train = res(temp(1: 80), 8)';
M = size(P_train, 2);

P_test = res(temp(81: end), 1: 7)';
T_test = res(temp(81: end), 8)';
N = size(P_test, 2);

%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
p_train =  double(reshape(P_train, 7, 1, 1, M));
p_test  =  double(reshape(P_test , 7, 1, 1, N));
t_train =  double(T_train)';
t_test  =  double(T_test )';

%%  构造网络结构
layers = [
 imageInputLayer([7, 1, 1])      % 输入层 输入数据规模[7, 1, 1]
 
 convolution2dLayer([3, 1], 16)  % 卷积核大小 3*1 生成16张特征图
 batchNormalizationLayer         % 批归一化层
 reluLayer                       % Relu激活层
 
 convolution2dLayer([3, 1], 32)  % 卷积核大小 3*1 生成32张特征图
 batchNormalizationLayer         % 批归一化层
 reluLayer                       % Relu激活层

 dropoutLayer(0.2)               % Dropout层
 fullyConnectedLayer(1)          % 全连接层
 regressionLayer];               % 回归层

%%  参数设置
options = trainingOptions('sgdm', ...      % SGDM 梯度下降算法
    'MiniBatchSize', 30, ...               % 批大小,每次训练样本个数30
    'MaxEpochs', 800, ...                  % 最大训练次数 800
    'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子
    'LearnRateDropPeriod', 400, ...        % 经过400次训练后 学习率为 0.01 * 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  模型预测
T_sim1 = predict(net, p_train);
T_sim2 = predict(net, p_test );

%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  绘制网络分析图
analyzeNetwork(layers)

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])