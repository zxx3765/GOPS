function jcob = IFEKF_mea_jcob_fcn_simp(x_k)
dt = 0.001;
jcob = [2*cos(x_k(5)*dt),-1,0,0,-dt*2*x_k(1)*sin(x_k(5)*dt);];

end