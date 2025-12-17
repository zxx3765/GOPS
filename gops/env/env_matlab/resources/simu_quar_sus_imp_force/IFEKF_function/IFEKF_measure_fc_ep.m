%% 此脚本用于创建基于IFEKF的状态转移函数


% 定义非线性函数
function y_k_1 = IFEKF_measure_fcn(x_k)
 

y_k_1 = x_k(5) * x_k(1) - x_k(2) - x_k(6)* x_k(3) + x_k(6) * x_k(4);


end