%% 此脚本用于创建基于IFEKF的状态转移函数


% 定义非线性函数
function y_k_1 = IFEKF_measure_fcn_simp_3(x_k)
 
dt = 0.001;
y_k_1 = 2*cos(x_k(3)*dt) * x_k(1) - x_k(2) ;


end