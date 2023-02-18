function [sol, val] = gabpEval(sol, ~)

%%  解码适应度值
val = gadecod(sol);