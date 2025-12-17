%% 此脚本用于创建基于IFEKF的状态转移函数


% 定义非线性函数
function x_k_1 = IFEKF_tfcn_simp(x_k)
x_k_1 = zeros(5,1);
dt = 0.001;
x_k_1(1) = 2*cos(x_k(5)*dt) * x_k(1) - x_k(2) ;
x_k_1(2) = x_k(1);
x_k_1(3) = 2*cos(x_k(5)*dt) * x_k(3) - x_k(4);
x_k_1(4) = x_k(3);
x_k_1(5) = x_k(5);
 
end