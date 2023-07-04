%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');

%%  划分训练集和测试集
temp = randperm(357);

P_train = res(temp(1: 240), 1: 12)';
T_train = res(temp(1: 240), 13)';
M = size(P_train, 2);

P_test = res(temp(241: end), 1: 12)';
T_test = res(temp(241: end), 13)';
N = size(P_test, 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

t_train = categorical(T_train)';
t_test  = categorical(T_test )';

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
P_train =  double(reshape(P_train, 12, 1, 1, M));
P_test  =  double(reshape(P_test , 12, 1, 1, N));

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1} = P_test( :, :, 1, i);
end

%%  创建网络
layers = [ ...
  sequenceInputLayer(12)               % 输入层
  
  lstmLayer(6, 'OutputMode', 'last')   % LSTM层
  reluLayer                            % Relu激活层
  
  fullyConnectedLayer(4)               % 全连接层
  softmaxLayer                         % 分类层
  classificationLayer];

%%  参数设置
options = trainingOptions('adam', ...       % Adam 梯度下降算法
    'MaxEpochs', 1000, ...                  % 最大迭代次数
    'InitialLearnRate', 0.01, ...           % 初始学习率
    'LearnRateSchedule', 'piecewise', ...   % 学习率下降
    'LearnRateDropFactor', 0.1, ...         % 学习率下降因子
    'LearnRateDropPeriod', 750, ...         % 经过 750 次训练后 学习率为 0.01 * 0.1
    'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
    'ValidationPatience', Inf, ...          % 关闭验证
    'Plots', 'training-progress', ...       % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  仿真预测
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

%%  数据反归一化
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  查看网络结构
analyzeNetwork(net)

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
