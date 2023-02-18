function newPop = normGeomSelect(oldPop, options)

% NormGeomSelect 是一个基于归一化几何分布的排序选择函数。

% newPop  - the new population selected from the oldPop
% oldPop  - the current population
% options - options to normGeomSelect [gen probability_of_selecting_best]

%%  交叉选择排序
q = options(2); 				    % 选择最佳的概率
e = size(oldPop, 2); 			    % xZome 的长度，即 numvars + fit
n = size(oldPop, 1);  		        % 种群数目
newPop = zeros(n, e); 		        % 为返回 pop 分配空间
fit = zeros(n, 1); 		            % 为选择概率分配空间
x = zeros(n,2); 			        % rank和id的排序列表
x(:, 1) = (n : -1 : 1)'; 	        % 要知道它是什么元素
[~, x(:, 2)] = sort(oldPop(:, e));  % 排序后获取索引

%%  相关参数
r = q / (1 - (1 - q) ^ n); 			            % 归一化分布，q 素数
fit(x(:, 2)) = r * (1 - q) .^ (x(:, 1) - 1); 	% 生成选择概率
fit = cumsum(fit); 			                    % 计算累积概率

%% 
rNums = sort(rand(n, 1)); 			            % 生成 n 个排序的随机数
fitIn = 1;                                      % 初始化循环控制
newIn = 1; 			                            % 初始化循环控制
while newIn <= n 				                % 获得 n 个新个体
  if(rNums(newIn) < fit(fitIn)) 		
    newPop(newIn, :) = oldPop(fitIn, :); 	    % 选择 fitIn 个人
    newIn = newIn + 1; 			                % 寻找下一个新人
  else
    fitIn = fitIn + 1; 			                % 着眼于下一个潜在选择
  end
end