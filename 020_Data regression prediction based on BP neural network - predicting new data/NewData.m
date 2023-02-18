%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取保存文件
load net.mat
load ps_input.mat
load ps_output.mat

%%  读取待预测数据
kes = xlsread('待预测数据.xlsx');

%%  数据转置
kes = kes';

%%  数据归一化
n_test = mapminmax('apply', kes, ps_input);

%%  仿真测试
t_sim3 = sim(net, n_test);

%%  数据反归一化
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%%  保存结果
xlswrite('预测结果.xlsx', T_sim3')