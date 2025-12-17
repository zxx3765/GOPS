function jcob = IFEKF_jcob_fcn_ep(x_k,epsilon)
dt = 0.001;
jcob = [cos(x_k(3)*dt), -sin(x_k(3)*dt), -dt*(x_k(1) * sin(x_k(3)*dt) + x_k(2)*cos(x_k(3)*dt)), 0;
        sin(x_k(3)*dt),  cos(x_k(3)*dt),  dt*(x_k(1) * cos(x_k(3)*dt) - x_k(2)*sin(x_k(3)*dt)), 0;
% jcob = [cos(x_k(3)), -sin(x_k(3)), -(x_k(1) * sin(x_k(3)) - x_k(2)*cos(x_k(3))), 0;
%          sin(x_k(3)),  cos(x_k(3)),  (x_k(1) * cos(x_k(3)) - x_k(2)*sin(x_k(3))), 0;
    0,0,1-epsilon,0;
    0,0,0,1];

end