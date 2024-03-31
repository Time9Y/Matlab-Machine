function model = initlssvm(X, Y, type, gam, sig2, kernel_type, preprocess)

%%  建立LSSVM模型
%  model = initlssvm(X, Y, type, gam, sig2, kernel, preprocess)
% 
%         model         : Object oriented representation of the LS-SVM model
%         X             : N x d matrix with the inputs of the training data
%         Y             : N x 1 vector with the outputs of the training data
%         type          : 'function estimation' ('f') or 'classifier' ('c')
%         kernel(*)     : Kernel type (by default 'RBF_kernel')
%         preprocess(*) : 'preprocess'(*) or 'original' 

%%  参数赋值
model.type    = type;
model.x_dim   = size(X, 2);
model.y_dim   = size(Y, 2);
model.nb_data = size(X, 1);

%%  初始化核函数参数
try 
    model.kernel_type = kernel_type;
catch 
    model.kernel_type = 'RBF_kernel';
end

%%  初始化归一化参数
try 
    model.preprocess = preprocess; 
catch 
    model.preprocess = 'preprocess';
end

%%  判断预处理状态
if  model.preprocess(1) == 'p'
    model.prestatus = 'changed';
else
    model.prestatus = 'ok'; 
end

%%  初始化数据点选择器
model.xtrain = X;
model.ytrain = Y;
model.selector = 1 : model.nb_data;

%%  初始化正则化项参数和核函数参数
if(gam <= 0)
    error('gam must be larger then 0');
end
model.gam = gam;

if sig2 <= 0
    model.kernel_pars = (model.x_dim);
else
    model.kernel_pars = sig2;
end

%%  动态模型参数设置(nar模式--尚未用到)
model.x_delays = 0;
model.y_delays = 0;
model.steps = 1;

%%  潜在变量？
model.latent = 'no';

%%  设置编码类型
model.code = 'original';

%%  设置编码方式
try 
    model.codetype = codetype; 
catch 
    model.codetype ='none';
end

%%  预处理步骤
model = prelssvm(model);

%%  模型状态
model.status = 'changed';

%%  设置权重
model.weights = [];
