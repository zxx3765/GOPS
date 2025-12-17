function jcob = IFEKF_mea_jcob_fcn_amp(x_k)
dt = 0.001;
jcob = [x_k(4)*cos(x_k(3)*dt), -x_k(4)*sin(x_k(3)*dt), -x_k(4)*dt*(x_k(1) * sin(x_k(3)*dt) + x_k(2)*cos(x_k(3)*dt)), x_k(1)*cos(x_k(3)*dt) - x_k(2)*sin(x_k(3)*dt);];

end