%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
result = xlsread('���ݼ�.xlsx');

%%  ���ݷ���
num_samples = length(result);  % ��������
or_dim = size(result, 2);      % ԭʼ���� + �����Ŀ
kim = 12;                      % ��ʱ������kim����ʷ������Ϊ�Ա�����
zim =  1;                      % ��zim��ʱ������Ԥ��

%%  �������ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1, :), 1, kim * or_dim), result(i + kim + zim - 1, end)];
end

%%  ����ѵ�����Ͳ��Լ�
temp = 1: 1: 1488;

P_train = res(temp(1: 1000), 1: end - 1)';
T_train = res(temp(1: 1000), end)';
M = size(P_train, 2);

P_test = res(temp(1001: end), 1: end - 1)';
T_test = res(temp(1001: end), end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ��
% ������ƽ�̳�1ά������һ�ִ���ʽ
% Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
% ����Ӧ��ʼ�պ���������ݽṹ����һ��
P_train =  double(reshape(P_train, 5, 12, 1, M));
P_test  =  double(reshape(P_test , 5, 12, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  ���ݸ�ʽת��
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  ����ģ��
layers = [
    sequenceInputLayer(5)               % ���������
    
    lstmLayer(10, 'OutputMode', 'last') % LSTM��
    
    fullyConnectedLayer(1)              % ȫ���Ӳ�
    regressionLayer];                   % �ع��
 
%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 500, ...                  % ���������� 500
    'MiniBatchSize', 128, ...              % ������С 
    'InitialLearnRate', 1e-3, ...          % ��ʼѧϰ��Ϊ0.001
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 400, ...        % ���� 400 ��ѵ���� ѧϰ��Ϊ 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % �������ݼ�
    'Plots', 'training-progress', ...      % ��������
    'Verbose', false);

%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);

%%  ����Ԥ��
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ά��ת��
T_sim1 = T_sim1'; T_sim2 = T_sim2';

%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  �鿴����ṹ
analyzeNetwork(net)

%%  ��ͼ
figure
plot(1: M, T_train, 'r-', 1: M, T_sim1, 'b-', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-', 1: N, T_sim2, 'b-', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M;
mae2 = sum(abs(T_sim2 - T_test )) ./ N;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

% MBE
mbe1 = sum(T_sim1 - T_train) ./ M;
mbe2 = sum(T_sim2 - T_test ) ./ N;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])

% MSE
mse1 = mean((T_sim1 - T_train).^2);
mse2 = mean((T_sim2 - T_test ).^2);

disp(['ѵ�������ݵ�MSEΪ��', num2str(mse1)])
disp(['���Լ����ݵ�MSEΪ��', num2str(mse2)])

% MAPE
mape1 = mean(abs((T_train - T_sim1) ./ T_train)) * 100;
mape2 = mean(abs((T_test  - T_sim2) ./ T_test )) * 100;

disp(['ѵ�������ݵ�MAPEΪ��', num2str(mape1), '%'])
disp(['���Լ����ݵ�MAPEΪ��', num2str(mape2), '%'])

%%  ����ɢ��ͼ
sz = 25;
c = 'b';

% ѵ����ɢ��ͼ
figure
scatter(T_train, T_sim1, sz, c)
hold on

min_val_train = min([T_train, T_sim1]);
max_val_train = max([T_train, T_sim1]);
plot([min_val_train, max_val_train], [min_val_train, max_val_train], '--k', 'LineWidth', 1.0)

xlabel('ѵ������ʵֵ');
ylabel('ѵ����Ԥ��ֵ');
xlim([min_val_train, max_val_train])
ylim([min_val_train, max_val_train])
title('ѵ����Ԥ��ֵ vs. ѵ������ʵֵ')
hold off

% ���Լ�ɢ��ͼ
figure
scatter(T_test, T_sim2, sz, c)
hold on

min_val_test = min([T_test, T_sim2]);
max_val_test = max([T_test, T_sim2]);
plot([min_val_test, max_val_test], [min_val_test, max_val_test], '--k', 'LineWidth', 1.0)

xlabel('���Լ���ʵֵ');
ylabel('���Լ�Ԥ��ֵ');
xlim([min_val_test, max_val_test])
ylim([min_val_test, max_val_test])
title('���Լ�Ԥ��ֵ vs. ���Լ���ʵֵ')
hold off