function [bestCVmse, bestc, bestg, pso_option] = psoSVMcgForRegress(train_label, train, pso_option)

%%  版权声明
% by faruto
% Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
% last modified 2010.01.17
% Super Moderator @ www.ilovematlab.cn

%%  若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%%  参数初始化
if nargin == 2
    pso_option = struct('c1', 1.5, 'c2', 1.7, 'maxgen', 20, 'sizepop', 20, ...
                    'k', 0.6, 'wV', 1, 'wP', 1, 'v',5, ...
                     'popcmax', 10^2, 'popcmin', 10^(-1), 'popgmax', 10^3, 'popgmin', 10^(-2));
end

%%  速度最大值设置
Vcmax = pso_option.k * pso_option.popcmax;
Vcmin = -Vcmax ;

Vgmax = pso_option.k * pso_option.popgmax;
Vgmin = -Vgmax ;

%%  设置误差阈值
eps = 10^(-10);

%%  产生初始粒子和速度
for i = 1 : pso_option.sizepop
    
    % 随机产生种群和速度
    pop(i, 1) = (pso_option.popcmax - pso_option.popcmin) * rand + pso_option.popcmin;  
    pop(i, 2) = (pso_option.popgmax - pso_option.popgmin) * rand + pso_option.popgmin;
    V(i, 1)   = Vcmax * rands(1, 1);  
    V(i, 2)   = Vgmax * rands(1, 1);
    
    % 计算初始适应度
    cmd = [' -v ', num2str(pso_option.v), ' -c ', num2str(pop(i, 1)), ' -g ', ...
        num2str(pop(i, 2)), ' -s 3 -p 0.1'];

    fitness(i) = svmtrain(train_label, train, cmd);

end

%%  初始化极值和极值点
[global_fitness, bestindex] = min(fitness);   % 全局极值
local_fitness = fitness;                      % 个体极值初始化
global_x = pop(bestindex, :);                 % 全局极值点
local_x = pop;                                % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1, pso_option.maxgen); 

%%  迭代寻优
for i = 1 : pso_option.maxgen
    for j = 1 : pso_option.sizepop
        
        % 速度更新        
        V(j, :) = pso_option.wV * V(j, :) + pso_option.c1 * rand * (local_x(j, :) - pop(j, :)) + ...
            pso_option.c2 * rand * (global_x - pop(j, :));

        if V(j, 1) > Vcmax
           V(j, 1) = Vcmax;
        end

        if V(j, 1) < Vcmin
           V(j, 1) = Vcmin;
        end

        if V(j, 2) > Vgmax
           V(j, 2) = Vgmax;
        end

        if V(j, 2) < Vgmin
           V(j, 2) = Vgmin;
        end
        
        % 种群更新
        pop(j, :) = pop(j, :) + pso_option.wP * V(j, :);
        
        if pop(j, 1) > pso_option.popcmax
           pop(j, 1) = pso_option.popcmax;
        end
        
        if pop(j, 1) < pso_option.popcmin
           pop(j, 1) = pso_option.popcmin;
        end
        
        if pop(j, 2) > pso_option.popgmax
           pop(j, 2) = pso_option.popgmax;
        end
        
        if pop(j, 2) < pso_option.popgmin
           pop(j, 2) = pso_option.popgmin;
        end
        
        % 自适应粒子变异
        if rand > 0.5
            k = ceil(2 * rand);

            if k == 1
               pop(j, k) = (20 - 1) * rand + 1;
            end

            if k == 2
               pop(j, k) = (pso_option.popgmax - pso_option.popgmin) * rand + pso_option.popgmin;
            end     

        end
        
        % 适应度值
        cmd = [' -v ', num2str(pso_option.v), ' -c ', num2str(pop(j, 1)), ' -g ', ...
            num2str(pop(j, 2)), ' -s 3 -p 0.1'];

        fitness(j) = svmtrain(train_label, train, cmd);
        
        % 个体最优更新
        if fitness(j) < local_fitness(j)
           local_x(j, :) = pop(j, :);
           local_fitness(j) = fitness(j);
        end

        if fitness(j) == local_fitness(j) && pop(j, 1) < local_x(j, 1)
           local_x(j, :) = pop(j, :);
           local_fitness(j) = fitness(j);
        end        
        
        % 群体最优更新
        if fitness(j) < global_fitness
           global_x = pop(j, :);
           global_fitness = fitness(j);
        end

        if abs(fitness(j) - global_fitness) <= eps && pop(j,1) < global_x(1)
           global_x = pop(j, :);
           global_fitness = fitness(j);
        end
        
    end
    
    fit_gen(i) = global_fitness;    
    avgfitness_gen(i) = sum(fitness) / pso_option.sizepop;
end

%%  适应度曲线
figure
plot(1 : length(fit_gen), fit_gen, 'b-', 'LineWidth', 1.5);
title('最佳适应度曲线', 'FontSize', 13);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度', 'FontSize', 10);
xlim([1, length(fit_gen)])
grid on

%%  赋值
bestc = global_x(1);
bestg = global_x(2);
bestCVmse = fit_gen(pso_option.maxgen);
