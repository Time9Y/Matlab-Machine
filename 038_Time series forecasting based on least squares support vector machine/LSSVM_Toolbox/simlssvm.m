function [Y, Yl, model] = simlssvm(model, Xt, A3)
%%  前向计算

% The matrix Xt represents the points one wants to predict. The
% first cell contains all arguments needed for defining the LS-SVM
% (see also trainlssvm, initlssvm). The second cell contains the
% results of training this LS-SVM model. The cell syntax allows for
% flexible and consistent default handling.

%        [Yt, Zt, model] = simlssvm(model, Xt)
%         Yt       : Nt x m matrix with predicted output of test data
%         Zt(*)    : Nt x m matrix with predicted latent variables of a classifier
%         model(*) : Object oriented representation of the LS-SVM model
%         model    : Object oriented representation of the LS-SVM model
%         Xt       : Nt x d matrix with the inputs of the test data

%%  判断模式
if iscell(model)

    % 初始化建立模型
    iscell_model = 1;
    model = initlssvm(model{:});

    % 给定权重
    if iscell(Xt)
        model.alpha = Xt{1};
        model.b = Xt{2};
        model.status = 'trained';

        try
            Xt = A3;
        catch
        end

    end
    Yt = [];

else

    %%  判断模式
    iscell_model = 0;
    Yt =[];
end

%%  检查维度是否对应
if size(Xt, 2) ~= model.x_dim
    error('dimensions of new datapoints Xt not equal to trainingsset...');
end

if ~isempty(Yt) && size(Yt, 2) ~= model.y_dim
    error('dimensions of new targetpoints Yt not equal to trainingsset...');
end

%%  预处理数据 
if model.preprocess(1) == 'p'
    Xt = prelssvm(model, Xt, Yt);
end

%%  判断是否训练
if  model.status(1) ~= 't' 
    warning('Model is not trained --> training now...')
    model = trainlssvm(model);
end

%%  判断是否多维输出
if model.y_dim > 1
    if (length(model.kernel_type) > 1 || size(model.kernel_pars, 2) > 1 || size(model.gam, 2) == model.y_dim)
        [Y, Yl] = simmultidimoutput(model, Xt);
        if  iscell_model
            model = Yl; 
        end
        return
    end
end

%%  使用 MATLAB 实现模拟模型
bz = 3000;
N = size(Xt, 1);
Y = zeros(N, 1);
NRofBlocks = floor(N / bz);
modu = N - NRofBlocks * bz;

for i = 1 : NRofBlocks
    indb = (i - 1) * bz + 1 : i * bz;
    Y(indb, :) = simFct(model, Xt(indb, :));
end

if modu ~= 0
    indb = NRofBlocks * bz + 1 : NRofBlocks * bz + modu;
    Y(indb, :) = simFct(model, Xt(indb, :));
end

%%  分类
Yl = Y;
if (model.type(1) == 'c' && strcmp(model.latent, 'no'))
    Y = 2 * (Y > 0) - 1;
end

%%  后期处理 
if (model.preprocess(1) == 'p' && ~(model.type(1) == 'c' && strcmp(model.latent, 'yes')))
    [~, Y] = postlssvm(model, [], Y);
end

%%  多分类解码
if (model.type(1) == 'c' && ~strcmpi(model.codetype, 'none') && ~strcmpi(model.code, 'original'))
    Y = codelssvm(model, Y);
end

%%  仿真模拟函数
function Y = simFct(model, X)
model.selector = ~isnan(model.ytrain);

kx = kernel_matrix(model.xtrain(model.selector, 1 : model.x_dim), model.kernel_type, ...
    model.kernel_pars, X);

Y = kx' * model.alpha(model.selector, 1 : model.y_dim) + ones(size(kx, 2), 1) * ...
    model.b(:, 1 : model.y_dim);

%%  多维输出函数
function [Yt, Yl] = simmultidimoutput(model, Xt)
Yt = []; 
Yl = [];

for d = 1 : model.y_dim

    try
        gam = model.gam(:, d);
    catch
        gam = model.gam;
    end

    try
        sig2 = model.kernel_pars(:, d);
    catch
        sig2 = model.kernel_pars;
    end

    try
        kernel = model.kernel_type{d};
    catch
        kernel = model.kernel_type;
    end

    [Ytn, Yln] = simlssvm({model.xtrain, model.ytrain(:, d), model.type, gam, ...
        sig2, kernel, 'original'}, {model.alpha(:, d), model.b(d)}, Xt);
    
    Yt = [Yt, Ytn]; 
    Yl = [Yl, Yln];
end

%%  后期处理
if (model.preprocess(1) == 'p' && ~(model.type(1) == 'c' && strcmp(model.latent, 'yes')))
    [~, Yt] = postlssvm(model, [], Yt);
end

%%  多分类解码
if (model.type(1) == 'c' && ~strcmpi(model.codetype, 'none' ) && ~strcmpi(model.code, 'original'))
    Yt = codelssvm(model, Yt);
end
