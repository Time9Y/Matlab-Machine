function omega = kernel_matrix(p_train, kernel_type, kernel_pars, p_test)

%%  样本个数
nb_data = size(p_train, 1);

%%  径向基核函数
if strcmp(kernel_type, 'RBF_kernel')
    if nargin < 4
        XXh   = sum(p_train.^2, 2) * ones(1, nb_data);
        omega = XXh + XXh' - 2 * (p_train * p_train');
        omega = exp(- omega ./ kernel_pars(1));
    else
        XXh1  = sum(p_train.^2, 2) * ones(1, size(p_test, 1));
        XXh2  = sum(p_test.^2, 2) * ones(1, nb_data);
        omega = XXh1 + XXh2' - 2 * p_train * p_test';
        omega = exp(- omega ./ kernel_pars(1));
    end
    
%%  线性核函数
elseif strcmp(kernel_type, 'lin_kernel')
    if nargin < 4
        omega = p_train * p_train';
    else
        omega = p_train * p_test';
    end

%%  多项式拟合核函数
elseif strcmp(kernel_type, 'poly_kernel')
    if nargin < 4
        omega = (p_train * p_train' + kernel_pars(1)) .^ kernel_pars(2);
    else
        omega = (p_train * p_test' + kernel_pars(1)) .^ kernel_pars(2);
    end
    
%%  小波核函数
elseif strcmp(kernel_type, 'wav_kernel')
    if nargin < 4
        XXh    = sum(p_train.^2, 2) * ones(1, nb_data);
        omega  = XXh + XXh' - 2 * (p_train * p_train');
        
        XXh1   = sum(p_train, 2) * ones(1, nb_data);
        omega1 = XXh1 - XXh1';
        omega  = cos(kernel_pars(3) * omega1 ./ kernel_pars(2)) .* exp(- omega ./ kernel_pars(1));
        
    else
        XXh1   = sum(p_train.^2, 2) * ones(1, size(p_test, 1));
        XXh2   = sum(p_test.^2, 2) * ones(1, nb_data);
        omega  = XXh1 + XXh2' - 2 * (p_train * p_test');
        
        XXh11  = sum(p_train, 2) * ones(1, size(p_test,1));
        XXh22  = sum(p_test, 2) * ones(1, nb_data);
        omega1 = XXh11 - XXh22';
        
        omega  = cos(kernel_pars(3) * omega1 ./ kernel_pars(2)) .* exp(-omega ./ kernel_pars(1));
    end
end
