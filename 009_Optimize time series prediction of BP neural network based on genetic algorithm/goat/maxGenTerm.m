function done = maxGenTerm(ops, ~, ~)
% 返回 1，即当达到 maximal_generation 时终止 GA。
%
% ops    - a vector of options [current_gen maximum_generation]
% bPop   - a matrix of best solutions [generation_found solution_string]
% endPop - the current generation of solutions

%%  
currentGen = ops(1);
maxGen     = ops(2);
done       = currentGen >= maxGen; 