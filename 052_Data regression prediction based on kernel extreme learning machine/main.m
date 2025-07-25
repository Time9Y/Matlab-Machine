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
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  训练模型
RC = 40;                % 正则化系数
Kernel_para = 1.0;      % 核函数参数
[t_sim1, t_sim2] = elm_kernel(p_train, t_train, p_test, RC, 'RBF_kernel', Kernel_para);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

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