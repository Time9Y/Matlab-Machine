function [t_sim1, t_sim2] = elm_kernel(p_train, t_train, p_test, Regularization_coefficient, Kernel_type, Kernel_para)

%%  参数介绍
% Input:
% Elm_Type                    - 0 for regression; 1 for (both binary and multi-classes) classification
% Regularization_coefficient  - Regularization coefficient C
% Kernel_type                 - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
% Kernel_para                 - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...

%%  正则化系数
C = Regularization_coefficient;

%%  得到输出权重
n = size(t_train, 2);
Omega_train = kernel_matrix(p_train', Kernel_type, Kernel_para);
OutputWeight = ((Omega_train + speye(n) / C) \ (t_train')); 

%%  训练集输出
t_sim1 = (Omega_train * OutputWeight)';                          

%%  测试集输出
Omega_test = kernel_matrix(p_train', Kernel_type, Kernel_para, p_test');
t_sim2 = (Omega_test' * OutputWeight)';                           