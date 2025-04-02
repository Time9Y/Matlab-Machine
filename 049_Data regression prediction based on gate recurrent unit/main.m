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

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
P_train =  double(reshape(P_train, 7, 1, 1, M));
P_test  =  double(reshape(P_test , 7, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  创建模型
layers = [
    sequenceInputLayer(7)                 % 建立输入层
    
    gruLayer(10, 'OutputMode', 'last')    % GRU层
    reluLayer                             % Relu激活层
    
    fullyConnectedLayer(1)                % 全连接层
    regressionLayer];                     % 回归层
 
%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 1500, ...                 % 最大训练次数 1500
    'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子 0.5
    'LearnRateDropPeriod', 400, ...        % 每经过400次训练后 学习率为 0.01*0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%%  训练模型
[net, Loss] = trainNetwork(p_train, t_train, layers, options);

%%  仿真预测
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  维度转置
T_sim1 = T_sim1'; T_sim2 = T_sim2';

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  查看网络结构
analyzeNetwork(net)

%%  损失函数曲线
figure
subplot(2, 1, 1)
plot(1 : length(Loss.TrainingRMSE), Loss.TrainingRMSE, 'r-', 'LineWidth', 1)
xlabel('迭代次数')
ylabel('均方根误差')
legend('训练集均方根误差')
title ('训练集均方根误差曲线')
grid
    
subplot(2, 1, 2)
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, 'b-', 'LineWidth', 1)
xlabel('迭代次数')
ylabel('损失函数')
legend('训练集损失值')
title ('训练集损失函数曲线')
grid

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
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M;
mae2 = sum(abs(T_sim2 - T_test )) ./ N;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

% MBE
mbe1 = sum(T_sim1 - T_train) ./ M;
mbe2 = sum(T_sim2 - T_test ) ./ N;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

% MSE
mse1 = mean((T_sim1 - T_train).^2);
mse2 = mean((T_sim2 - T_test ).^2);

disp(['训练集数据的MSE为：', num2str(mse1)])
disp(['测试集数据的MSE为：', num2str(mse2)])

% MAPE
mape1 = mean(abs((T_train - T_sim1) ./ T_train)) * 100;
mape2 = mean(abs((T_test  - T_sim2) ./ T_test )) * 100;

disp(['训练集数据的MAPE为：', num2str(mape1), '%'])
disp(['测试集数据的MAPE为：', num2str(mape2), '%'])

%%  绘制散点图
sz = 25;
c = 'b';

% 训练集散点图
figure
scatter(T_train, T_sim1, sz, c)
hold on

min_val_train = min([T_train, T_sim1]);
max_val_train = max([T_train, T_sim1]);
plot([min_val_train, max_val_train], [min_val_train, max_val_train], '--k', 'LineWidth', 1.0)

xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min_val_train, max_val_train])
ylim([min_val_train, max_val_train])
title('训练集预测值 vs. 训练集真实值')
hold off

% 测试集散点图
figure
scatter(T_test, T_sim2, sz, c)
hold on

min_val_test = min([T_test, T_sim2]);
max_val_test = max([T_test, T_sim2]);
plot([min_val_test, max_val_test], [min_val_test, max_val_test], '--k', 'LineWidth', 1.0)

xlabel('测试集真实值');
ylabel('测试集预测值');
xlim([min_val_test, max_val_test])
ylim([min_val_test, max_val_test])
title('测试集预测值 vs. 测试集真实值')
hold off