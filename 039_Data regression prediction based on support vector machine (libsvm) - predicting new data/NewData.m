%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取保存网络和变量
load save_vars\model.mat
load save_vars\ps_input.mat
load save_vars\ps_output.mat

%%  读取待预测数据
kes = xlsread('待预测数据.xlsx');

%%  数据转置
kes = kes';

%%  数据归一化
n_test = mapminmax('apply', kes, ps_input);

%%  数据转置
n_test = n_test';
out_n_test = zeros(1, size(kes, 2))';

%%  仿真测试
t_sim3 = svmpredict(out_n_test, n_test, model);

%%  反归一化
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%%  保存结果
xlswrite('预测结果.xlsx', T_sim3)

%%  打印结果
clc
disp("预测结果保存完毕！")