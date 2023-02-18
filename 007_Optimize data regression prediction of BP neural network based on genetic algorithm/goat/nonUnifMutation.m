function parent = nonUnifMutation(parent, bounds, Ops)

%%  非均匀突变基于非均匀概率分布改变父代的参数之一
% parent  - the first parent ( [solution string function value] )
% bounds  - the bounds matrix for the solution space
% Ops     - Options for nonUnifMutate[gen #NonUnifMutations maxGen b]

%%  相关参数设置
cg = Ops(1); 				              % 当前这一代
mg = Ops(3);                              % 最大代数
bm = Ops(4);                              % 形状参数
numVar = size(parent, 2) - 1; 	          % 获取变量个数
mPoint = round(rand * (numVar - 1)) + 1;  % 选择一个变量从 1 到变量数随机变化
md = round(rand); 			              % 选择突变方向
if md 					                  % 向上限突变
  newValue = parent(mPoint) + delta(cg, mg, bounds(mPoint, 2) - parent(mPoint), bm);
else 					                  % 向下限突变
  newValue = parent(mPoint) - delta(cg, mg, parent(mPoint) - bounds(mPoint, 1), bm);
end
parent(mPoint) = newValue; 		          % 产生子代