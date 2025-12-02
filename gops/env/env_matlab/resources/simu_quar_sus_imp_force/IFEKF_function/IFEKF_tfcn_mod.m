%% 此脚本用于创建基于IFEKF的状态转移函数


% 定义非线性函数
function x_k_1 = IFEKF_tfcn_mod(x_k)
dt = 0.001;
x_k_1 = zeros(4,1);
x_k_1(1) = (x_k(1) * cos(x_k(3)*dt) - x_k(2)*sin(x_k(3)*dt));
x_k_1(2) = (x_k(1) * sin(x_k(3)*dt) + x_k(2)*cos(x_k(3)*dt));
x_k_1(3) = x_k(3);
x_k_1(4) = x_k(4);
 
end