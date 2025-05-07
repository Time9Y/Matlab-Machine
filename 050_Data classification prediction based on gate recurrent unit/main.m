%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = xlsread('���ݼ�.xlsx');

%%  ����ѵ�����Ͳ��Լ�
temp = randperm(357);

P_train = res(temp(1: 240), 1: 12)';
T_train = res(temp(1: 240), 13)';
M = size(P_train, 2);

P_test = res(temp(241: end), 1: 12)';
T_test = res(temp(241: end), 13)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
P_train =  double(reshape(P_train, 12, 1, 1, M));
P_test  =  double(reshape(P_test , 12, 1, 1, N));

%%  ���ݸ�ʽת��
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  ��������
layers = [ ...
  sequenceInputLayer(12)               % �����
  
  gruLayer(7, 'OutputMode', 'last')    % GRU��
  reluLayer                            % Relu�����
  
  fullyConnectedLayer(4)               % ȫ���Ӳ�
  softmaxLayer                         % �����
  classificationLayer];

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 800, ...                  % ���ѵ������ 800
    'MiniBatchSize', 128, ...              % ����С 128
    'InitialLearnRate', 1e-2, ...          % ��ʼѧϰ��Ϊ 0.01
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.5, ...        % ѧϰ���½����� 0.5
    'LearnRateDropPeriod', 200, ...        % ÿ���� 200 ��ѵ���� ѧϰ�� * 0.5
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'ValidationPatience', Inf, ...         % �ر���֤
    'Plots', 'training-progress', ...      % ��������
    'Verbose', false);

%%  ѵ��ģ��
[net, Loss] = trainNetwork(p_train, t_train, layers, options);

%%  ����Ԥ��
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

%%  ���ݷ���һ��
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  ��������
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  ��ʧ��������
figure
subplot(2, 1, 1)
plot(1 : length(Loss.TrainingAccuracy), Loss.TrainingAccuracy, 'r-', 'LineWidth', 1)
xlabel('��������')
ylabel('׼ȷ��')
legend('ѵ����׼ȷ��')
title ('ѵ����׼ȷ������')
grid
    
subplot(2, 1, 2)
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, 'b-', 'LineWidth', 1)
xlabel('��������')
ylabel('��ʧ����')
legend('ѵ������ʧֵ')
title ('ѵ������ʧ��������')
grid

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
grid

%%  ��������
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