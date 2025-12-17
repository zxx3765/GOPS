function jcob = IFEKF_mea_jcob_fcn_mod(x_k)
dt = 0.001;
jcob = [cos(x_k(3)*dt), -sin(x_k(3)*dt), -dt*(x_k(1) * sin(x_k(3)*dt) + x_k(2)*cos(x_k(3)*dt)), 0;];

end