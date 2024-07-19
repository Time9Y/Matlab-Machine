function [model, b, X, Y]  = trainlssvm(model, X, Y)

%%  训练模型
% >> model = trainlssvm(model)
% type can be 'classifier' or 'function estimation' (these strings
% can be abbreviated into 'c' or 'f', respectively). X and Y are
% matrices holding the training input and output data. The i-th
% data point is represented by the i-th row X(i,:) and Y(i,:). gam
% is the regularization parameter: for gam low minimizing of the
% complexity of the model is emphasized, for gam high, good fitting
% of the training data points is stressed. kernel_par is the
% parameter of the kernel; in the common case of an RBF kernel, a
% large sig2 indicates a stronger smoothing. The kernel_type
% indicates the function that is called to compute the kernel value
% (by default RBF_kernel). Other kernels can be used for example:
%
% The kernel parameter(s) are passed as a row vector, in the case
% no kernel parameter is needed, pass the empty vector!
%
% The training can either be proceeded by the preprocessing
% function ('preprocess') (by default) or not ('original'). The
% training calls the preprocessing (prelssvm, postlssvm) and the
% encoder (codelssvm) if appropiate.
%
% In the remainder of the text, the content of the cell determining
% the LS-SVM is given by {X,Y, type, gam, sig2}. However, the
% additional arguments in this cell can always be added in the
% calls.
%
% If one uses the object oriented interface (see also A.3.14), the training is done by
%
% >> model = trainlssvm(model)
% >> model = trainlssvm(model, X, Y)
%
% The status of the model checks whether a retraining is
% needed. The extra arguments X, Y allow to re-initialize the model
% with this new training data as long as its dimensions are the
% same as the old initiation.
%
% The training implementation:
%
%     * The Matlab implementation: a straightforward implementation
%     based on the matrix division '\' (lssvmMATLAB.m). 
%
%
% This implementation allows to train a multidimensional output
% problem. If each output uses the same kernel type, kernel
% parameters and regularization parameter, this is
% straightforward. If not so, one can specify the different types
% and/or parameters as a row vector in the appropriate
% argument. Each dimension will be trained with the corresponding
% column in this vector.

%   model = trainlssvm(model)
%         model          : Trained object oriented representation of the LS-SVM model
%         model          : Object oriented representation of the LS-SVM model
%         X(*)           : N x d matrix with the inputs of the training data
%         Y(*)           : N x 1 vector with the outputs of the training data
%         type(*)        : 'function estimation' ('f') or 'classifier' ('c')
%         gam(*)         : Regularization parameter
%         sig2(*)        : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%         kernel(*)      : Kernel type (by default 'RBF_kernel')
%         preprocess(*)  : 'preprocess'(*) or 'original'

%%  初始化模型
if iscell(model)
    model = initlssvm(model{:});
end

%%  判断模型训练状态
if model.status(1) == 't'
    if (nargout > 1)
        X = model.xtrain;
        Y = model.ytrain;
        b = model.b;
        model = model.alpha;
    end
    return
end

%%  控制输入
if ~((strcmp(model.kernel_type, 'RBF_kernel')  && length(model.kernel_pars) >= 1) ||...
     (strcmp(model.kernel_type, 'lin_kernel')  && length(model.kernel_pars) >= 0) ||...
     (strcmp(model.kernel_type, 'MLP_kernel')  && length(model.kernel_pars) >= 2) ||...
     (strcmp(model.kernel_type, 'poly_kernel') && length(model.kernel_pars) >= 1))
    
elseif (model.steps <= 0)
    error('steps must be larger then 0');

elseif (model.gam <= 0)
    error('gamma must be larger then 0');

elseif or(model.x_dim <= 0, model.y_dim <= 0)
    error('dimension of datapoints must be larger than 0');

end

%%  编码（分类才需要）
if model.code(1) == 'c'
    model = codelssvm(model);
end

%%  赋值
try
    if model.prestatus(1)=='c'
       changed = 1; 
    else 
       changed = 0;
    end
catch
    changed = 0;
end
        
if model.preprocess(1) == 'p' && changed
    model = prelssvm(model);

elseif model.preprocess(1) == 'o' && changed
    model = postlssvm(model);

end

%%  计时开始
tic;

%%  执行方式
if size(model.gam, 1) > 1
    model.implementation = 'MATLAB';
end

%%  递归调用输出维度预测
if model.y_dim > 1
    if (length(model.kernel_pars) == model.y_dim || size(model.gam, ...
            2) == model.y_dim || numel(model.kernel_type, 2) == model.y_dim)
        model = trainmultidimoutput(model);

        if (nargout > 1)
            X = model.xtrain;
            Y = model.ytrain;
            b = model.b;
            model = model.alpha;
        else
            model.duration = toc;
            model.status = 'trained';
        end

        return
    end
end

%%  
model = lssvmMATLAB(model);

if (nargout > 1)
    X = model.xtrain;
    Y = model.ytrain;
    b = model.b;
    model = model.alpha;
else
    model.duration = toc;
    model.status = 'trained';
end


%%  训练多输出模型
function model = trainmultidimoutput(model)

%%  初始化权重
model.alpha = zeros(model.nb_data, model.y_dim);
model.b = zeros(1, model.y_dim);

for d = 1 : model.y_dim

    try
        gam = model.gam(:, d);
    catch
        gam = model.gam(:);
    end

    try
        sig2 = model.kernel_pars(:, d);
    catch
        sig2 = model.kernel_pars(:);
    end

    try
        kernel = model.kernel_type{d};
    catch
        kernel=model.kernel_type;
    end

    [model.alpha(:, d), model.b(d)] = trainlssvm({model.xtrain, model.ytrain(:, d), ...
        model.type, gam, sig2, kernel, 'original'});
end

%%  输出
if (nargout > 1)
    model = model.alpha;
else
    model.duration = toc;
    model.status = 'trained';
end
