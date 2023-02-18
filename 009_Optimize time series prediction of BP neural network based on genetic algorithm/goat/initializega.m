function pop = initializega(num, bounds, evalFN, evalOps, options)

%%  种群初始化
%    initializega creates a matrix of random numbers with 
%    a number of rows equal to the populationSize and a number
%    columns equal to the number of rows in bounds plus 1 for
%    the f(x) value which is found by applying the evalFN.
%    This is used by the ga to create the population if it
%    is not supplied.
%
% pop            - the initial, evaluated, random population 
% populatoinSize - the size of the population, i.e. the number to create
% variableBounds - a matrix which contains the bounds of each variable, i.e.
%                  [var1_high var1_low; var2_high var2_low; ....]
% evalFN         - the evaluation fn, usually the name of the .m file for 
%                  evaluation
% evalOps        - any options to be passed to the eval function defaults []
% options        - options to the initialize function, ie. 
%                  [type prec] where eps is the epsilon value 
%                  and the second option is 1 for float and 0 for binary, 
%                  prec is the precision of the variables defaults [1e-6 1]

%%  参数初始化
if nargin < 5
  options = [1e-6, 1];
end
if nargin < 4
  evalOps = [];
end

%%  编码方式
if any(evalFN < 48)    % M文件
  if options(2) == 1   % 浮点数编码
    estr = ['x=pop(i,1); pop(i,xZomeLength)=', evalFN ';'];  
  else                 % 二进制编码
    estr = ['x=b2f(pop(i,:),bounds,bits); pop(i,xZomeLength)=', evalFN ';']; 
  end
else                   % 非M文件
  if options(2) == 1   % 浮点数编码
    estr = ['[ pop(i,:) pop(i,xZomeLength)]=' evalFN '(pop(i,:),[0 evalOps]);']; 
  else                 % 二进制编码
    estr = ['x=b2f(pop(i,:),bounds,bits);[x v]=' evalFN ...
	'(x,[0 evalOps]); pop(i,:)=[f2b(x,bounds,bits) v];'];  
  end
end

%%  参数设置 
numVars = size(bounds, 1); 		           % 变量数
rng     = (bounds(:, 2) - bounds(:, 1))';  % 可变范围

%%  编码方式
if options(2) == 1               % 二进制编码
  xZomeLength = numVars + 1; 	 % 字符串的长度是 numVar + fit
  pop = zeros(num, xZomeLength); % 分配新种群
  pop(:, 1 : numVars) = (ones(num, 1) * rng) .* (rand(num, numVars)) + ...
    (ones(num, 1) * bounds(:, 1)');
else                             % 浮点数编码
  bits = calcbits(bounds, options(1));
  pop = round(rand(num, sum(bits) + 1));
end

%%  运行文件
for i = 1 : num
  eval(estr);
end
