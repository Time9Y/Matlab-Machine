%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取模型
load model.mat
load ps_input.mat
load ps_output.mat

%%  导入数据
res = xlsread('需要预测的数据.xlsx');

%%  样本数目
M = size(res, 1);

%%  数据归一化
p_test = mapminmax('apply', res', ps_input);

%%  转置以适应模型
p_test = p_test';

%%  仿真测试
t_sim3 = regRF_predict(p_test, model);

%%  数据反归一化
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%%  保存结果
xlswrite('预测结果.xlsx', T_sim3);
