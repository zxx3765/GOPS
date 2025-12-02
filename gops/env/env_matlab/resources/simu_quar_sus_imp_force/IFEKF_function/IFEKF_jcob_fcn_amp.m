function jcob = IFEKF_jcob_fcn_amp(x_k,epsilon)
dt = 0.001;
jcob = [x_k(4)*cos(x_k(3)*dt), -x_k(4)*sin(x_k(3)*dt), -x_k(4)*dt*(x_k(1) * sin(x_k(3)*dt) + x_k(2)*cos(x_k(3)*dt)), x_k(1)*cos(x_k(3)*dt) - x_k(2)*sin(x_k(3)*dt);
        x_k(4)*sin(x_k(3)*dt),  x_k(4)*cos(x_k(3)*dt),  x_k(4)*dt*(x_k(1) * cos(x_k(3)*dt) - x_k(2)*sin(x_k(3)*dt)), x_k(1)*cos(x_k(3)*dt) + x_k(2)*sin(x_k(3)*dt);
% jcob = [cos(x_k(3)), -sin(x_k(3)), -(x_k(1) * sin(x_k(3)) - x_k(2)*cos(x_k(3))), 0;
%          sin(x_k(3)),  cos(x_k(3)),  (x_k(1) * cos(x_k(3)) - x_k(2)*sin(x_k(3))), 0;
    0,0,1-epsilon,0;
    0,0,0,1];

end