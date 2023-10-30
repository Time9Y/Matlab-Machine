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
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  参数设置
pso_option.c1      = 1.5;                    % c1:初始为1.5, pso参数局部搜索能力
pso_option.c2      = 1.7;                    % c2:初始为1.7, pso参数全局搜索能力
pso_option.maxgen  = 100;                    % maxgen:最大进化数量设置为100
pso_option.sizepop =  5;                     % sizepop:种群最大数量设置为5
pso_option.k  = 0.6;                         % 初始为0.6(k belongs to [0.1,1.0]),速率和x的关系(V = kX)
pso_option.wV = 1;                           % wV:初始为1(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
pso_option.wP = 1;                           % wP:初始为1,种群更新公式中速度前面的弹性系数
pso_option.v  = 3;                           % v:初始为3,SVM Cross Validation参数

pso_option.popcmax = 100;                    % popcmax:初始为100, SVM 参数c的变化的最大值.
pso_option.popcmin = 0.1;                    % popcmin:初始为0.1, SVM 参数c的变化的最小值.
pso_option.popgmax = 100;                    % popgmax:初始为100, SVM 参数g的变化的最大值.
pso_option.popgmin = 0.1;                    % popgmin:初始为0.1, SVM 参数c的变化的最小值.

%%  提取最佳参数c和g
[bestacc, bestc, bestg] = pso_svm_class(t_train, p_train, pso_option);

%%  建立模型
cmd = [' -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = svmtrain(t_train, p_train, cmd);

%%  仿真测试
T_sim1 = svmpredict(t_train, p_train, model);
T_sim2 = svmpredict(t_test , p_test , model);

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
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