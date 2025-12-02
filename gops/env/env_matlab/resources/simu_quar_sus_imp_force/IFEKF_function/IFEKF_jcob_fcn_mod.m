function jcob = IFEKF_jcob_fcn_mod(x_k)
dt = 0.001;
jcob = [cos(x_k(3)*dt), -sin(x_k(3)*dt), -dt*(x_k(1) * sin(x_k(3)*dt) - x_k(2)*cos(x_k(3)*dt)), 0;
        sin(x_k(3)*dt),  cos(x_k(3)*dt),  dt*(x_k(1) * cos(x_k(3)*dt) - x_k(2)*sin(x_k(3)*dt)), 0;
    0,0,1,0;
    0,0,0,1];

end