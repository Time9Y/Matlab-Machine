function [x, endPop, bPop, traceInfo] = ga(bounds, evalFN, evalOps, startPop, opts, ...
termFN, termOps, selectFN, selectOps, xOverFNs, xOverOps, mutFNs, mutOps)
                              
% Output Arguments:
%   x            - the best solution found during the course of the run
%   endPop       - the final population 
%   bPop         - a trace of the best population
%   traceInfo    - a matrix of best and means of the ga for each generation
%
% Input Arguments:
%   bounds       - a matrix of upper and lower bounds on the variables
%   evalFN       - the name of the evaluation .m function
%   evalOps      - options to pass to the evaluation function ([NULL])
%   startPop     - a matrix of solutions that can be initialized
%                  from initialize.m
%   opts         - [epsilon prob_ops display] change required to consider two 
%                  solutions different, prob_ops 0 if you want to apply the
%                  genetic operators probabilisticly to each solution, 1 if
%                  you are supplying a deterministic number of operator
%                  applications and display is 1 to output progress 0 for
%                  quiet. ([1e-6 1 0])
%   termFN       - name of the .m termination function (['maxGenTerm'])
%   termOps      - options string to be passed to the termination function
%                  ([100]).
%   selectFN     - name of the .m selection function (['normGeomSelect'])
%   selectOpts   - options string to be passed to select after
%                  select(pop,#,opts) ([0.08])
%   xOverFNS     - a string containing blank seperated names of Xover.m
%                  files (['arithXover heuristicXover simpleXover']) 
%   xOverOps     - A matrix of options to pass to Xover.m files with the
%                  first column being the number of that xOver to perform
%                  similiarly for mutation ([2 0;2 3;2 0])
%   mutFNs       - a string containing blank seperated names of mutation.m 
%                  files (['boundaryMutation multiNonUnifMutation ...
%                           nonUnifMutation unifMutation'])
%   mutOps       - A matrix of options to pass to Xover.m files with the
%                  first column being the number of that xOver to perform
%                  similiarly for mutation ([4 0 0;6 100 3;4 100 3;4 0 0])

%%  初始化参数
n = nargin;
if n < 2 || n == 6 || n == 10 || n == 12
  disp('Insufficient arguements') 
end

% 默认评估选项
if n < 3 
  evalOps = [];
end

% 默认参数
if n < 5
  opts = [1e-6, 1, 0];
end

% 默认参数
if isempty(opts)
  opts = [1e-6, 1, 0];
end

%%  判断是否为m文件
if any(evalFN < 48)
  % 浮点数编码 
  if opts(2) == 1
    e1str = ['x=c1; c1(xZomeLength)=', evalFN ';'];  
    e2str = ['x=c2; c2(xZomeLength)=', evalFN ';']; 
  % 二进制编码
  else
    e1str = ['x=b2f(endPop(j,:),bounds,bits); endPop(j,xZomeLength)=', evalFN ';'];
  end
else
  % 浮点数编码
  if opts(2) == 1
    e1str = ['[c1 c1(xZomeLength)]=' evalFN '(c1,[gen evalOps]);'];  
    e2str = ['[c2 c2(xZomeLength)]=' evalFN '(c2,[gen evalOps]);'];
  % 二进制编码
  else
    e1str=['x=b2f(endPop(j,:),bounds,bits);[x v]=' evalFN ...
	'(x,[gen evalOps]); endPop(j,:)=[f2b(x,bounds,bits) v];'];  
  end
end

%%  默认终止信息
if n < 6
  termOps = 100;
  termFN = 'maxGenTerm';
end

%%  默认变异信息
if n < 12
  % 浮点数编码
  if opts(2) == 1
  mutFNs = 'boundaryMutation multiNonUnifMutation nonUnifMutation unifMutation';
    mutOps = [4, 0, 0; 6, termOps(1), 3; 4, termOps(1), 3;4, 0, 0];
  % 二进制编码
  else
    mutFNs = 'binaryMutation';
    mutOps = 0.05;
  end
end

%%  默认交叉信息
if n < 10
  % 浮点数编码
  if opts(2) == 1
    xOverFNs = 'arithXover heuristicXover simpleXover';
    xOverOps = [2, 0; 2, 3; 2, 0];
  % 二进制编码
  else
    xOverFNs = 'simpleXover';
    xOverOps = 0.6;
  end
end

%%  仅默认选择选项，即轮盘赌。
if n < 9
  selectOps = [];
end

%%  默认选择信息
if n < 8
  selectFN = 'normGeomSelect';
  selectOps = 0.08;
end

%%  默认终止信息
if n < 6
  termOps = 100;
  termFN = 'maxGenTerm';
end

%%  没有定的初始种群
if n < 4
  startPop = [];
end

%%  随机生成种群
if isempty(startPop)
  startPop = initializega(80, bounds, evalFN, evalOps, opts(1: 2));
end

%%  二进制编码
if opts(2) == 0
  bits = calcbits(bounds, opts(1));
end

%%  参数设置
xOverFNs     = parse(xOverFNs);
mutFNs       = parse(mutFNs);
xZomeLength  = size(startPop, 2); 	          % xzome 的长度
numVar       = xZomeLength - 1; 	          % 变量数
popSize      = size(startPop,1); 	          % 种群人口个数
endPop       = zeros(popSize, xZomeLength);   % 第二种群矩阵
numXOvers    = size(xOverFNs, 1);             % Number of Crossover operators
numMuts      = size(mutFNs, 1); 		      % Number of Mutation operators
epsilon      = opts(1);                       % Threshold for two fittness to differ
oval         = max(startPop(:, xZomeLength)); % Best value in start pop
bFoundIn     = 1; 			                  % Number of times best has changed
done         = 0;                             % Done with simulated evolution
gen          = 1; 			                  % Current Generation Number
collectTrace = (nargout > 3); 		          % Should we collect info every gen
floatGA      = opts(2) == 1;                  % Probabilistic application of ops
display      = opts(3);                       % Display progress 

%%  精英模型
while(~done)
  [bval, bindx] = max(startPop(:, xZomeLength));            % Best of current pop
  best =  startPop(bindx, :);
  if collectTrace
    traceInfo(gen, 1) = gen; 		                        % current generation
    traceInfo(gen, 2) = startPop(bindx,  xZomeLength);      % Best fittness
    traceInfo(gen, 3) = mean(startPop(:, xZomeLength));     % Avg fittness
    traceInfo(gen, 4) = std(startPop(:,  xZomeLength)); 
  end
  
  %%  最佳解
  if ( (abs(bval - oval) > epsilon) || (gen==1))
    
    % 更新显示
    if display
      fprintf(1, '\n%d %f\n', gen, bval);          
    end

    % 更新种群矩阵
    if floatGA
      bPop(bFoundIn, :) = [gen, startPop(bindx, :)]; 
    else
      bPop(bFoundIn, :) = [gen, b2f(startPop(bindx, 1 : numVar), bounds, bits)...
	  startPop(bindx, xZomeLength)];
    end

    bFoundIn = bFoundIn + 1;                      % Update number of changes
    oval = bval;                                  % Update the best val
  else
    if display
      fprintf(1,'%d ',gen);	                      % Otherwise just update num gen
    end
  end
%%  选择种群
  endPop = feval(selectFN, startPop, [gen, selectOps]);
  
  % 以参数为操作数的模型运行
  if floatGA
    for i = 1 : numXOvers
      for j = 1 : xOverOps(i, 1)
          a = round(rand * (popSize - 1) + 1); 	     % Pick a parent
	      b = round(rand * (popSize - 1) + 1); 	     % Pick another parent
	      xN = deblank(xOverFNs(i, :)); 	         % Get the name of crossover function
	      [c1, c2] = feval(xN, endPop(a, :), endPop(b, :), bounds, [gen, xOverOps(i, :)]);

          % Make sure we created a new 
          if c1(1 : numVar) == endPop(a, (1 : numVar)) 
	         c1(xZomeLength) = endPop(a, xZomeLength);
	      elseif c1(1:numVar) == endPop(b, (1 : numVar))
	         c1(xZomeLength) = endPop(b, xZomeLength);
          else
             eval(e1str);
          end

          if c2(1 : numVar) == endPop(a, (1 : numVar))
	          c2(xZomeLength) = endPop(a, xZomeLength);
	      elseif c2(1 : numVar) == endPop(b, (1 : numVar))
	          c2(xZomeLength) = endPop(b, xZomeLength);
          else
	          eval(e2str);
          end

          endPop(a, :) = c1;
          endPop(b, :) = c2;
      end
    end

    for i = 1 : numMuts
      for j = 1 : mutOps(i, 1)
          a = round(rand * (popSize - 1) + 1);
          c1 = feval(deblank(mutFNs(i, :)), endPop(a, :), bounds, [gen, mutOps(i, :)]);
          if c1(1 : numVar) == endPop(a, (1 : numVar)) 
              c1(xZomeLength) = endPop(a, xZomeLength);
          else
              eval(e1str);
          end
          endPop(a, :) = c1;
      end
    end

%%  运行遗传算子的概率模型
  else 
    for i = 1 : numXOvers
        xN = deblank(xOverFNs(i, :));
        cp = find((rand(popSize, 1) < xOverOps(i, 1)) == 1);

        if rem(size(cp, 1), 2) 
            cp = cp(1 : (size(cp, 1) - 1)); 
        end
        cp = reshape(cp, size(cp, 1) / 2, 2);

        for j = 1 : size(cp, 1)
            a = cp(j, 1); 
            b = cp(j, 2); 
            [endPop(a, :), endPop(b, :)] = feval(xN, endPop(a, :), endPop(b, :), ...
                bounds, [gen, xOverOps(i, :)]);
        end
    end

    for i = 1 : numMuts
        mN = deblank(mutFNs(i, :));
        for j = 1 : popSize
            endPop(j, :) = feval(mN, endPop(j, :), bounds, [gen, mutOps(i, :)]);
            eval(e1str);
        end
    end

  end
  
  %  更新记录
  gen = gen + 1;
  done = feval(termFN, [gen, termOps], bPop, endPop); % See if the ga is done
  startPop = endPop; 			                      % Swap the populations
  [~, bindx] = min(startPop(:, xZomeLength));         % Keep the best solution
  startPop(bindx, :) = best; 		                  % replace it with the worst
  
end
[bval, bindx] = max(startPop(:, xZomeLength));

%%  显示结果
if display 
  fprintf(1, '\n%d %f\n', gen, bval);	  
end

%%  二进制编码
x = startPop(bindx, :);
if opts(2) == 0
  x = b2f(x, bounds,bits);
  bPop(bFoundIn, :) = [gen, b2f(startPop(bindx, 1 : numVar), bounds, bits)...
      startPop(bindx, xZomeLength)];
else
  bPop(bFoundIn, :) = [gen, startPop(bindx, :)];
end

%%  赋值
if collectTrace
  traceInfo(gen, 1) = gen; 		                      % 当前迭代次数
  traceInfo(gen, 2) = startPop(bindx, xZomeLength);   % 最佳适应度
  traceInfo(gen, 3) = mean(startPop(:, xZomeLength)); % 平均适应度
end