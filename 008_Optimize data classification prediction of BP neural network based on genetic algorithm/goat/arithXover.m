function [C1, C2] = arithXover(P1, P2, ~, ~)
%%  Arith 交叉采用两个父节点 P1、P2 并沿两个父节点形成的线执行插值。
% P1      - the first parent ( [solution string function value] )
% P2      - the second parent ( [solution string function value] )
% bounds  - the bounds matrix for the solution space
% Ops     - Options matrix for arith crossover [gen #ArithXovers]

%%  选择一个随机的混合量
a = rand;

%%  创建子代
C1 = P1 * a + P2 * (1 - a);
C2 = P1 * (1 - a) + P2 * a;