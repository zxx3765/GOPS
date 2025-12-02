function jcob = IFEKF_jcob_fcn_simp_3(x_k)
dt = 0.001;
jcob = [2*cos(x_k(3)*dt),-1,-dt*2*x_k(1)*sin(x_k(3)*dt);
    1,0,0;
    0,0,1;];

end