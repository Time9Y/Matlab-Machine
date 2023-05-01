function error = fun(pop, hiddennum, net, p_train, t_train)

%%  节点个数
inputnum  = size(p_train, 1);  % 输入层节点数
outputnum = size(t_train, 1);  % 输出层节点数

%%  提取权值和阈值
w1 = pop(1 : inputnum * hiddennum);
B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum);
B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

%%  网络赋值
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%%  网络训练
net = train(net, p_train, t_train);

%%  仿真测试
t_sim1 = sim(net, p_train);

%%  反归一化
T_sim1  = vec2ind(t_sim1 );
T_train = vec2ind(t_train);

%%  适应度值
error = 1 - sum(T_sim1 == T_train) / length(T_sim1);