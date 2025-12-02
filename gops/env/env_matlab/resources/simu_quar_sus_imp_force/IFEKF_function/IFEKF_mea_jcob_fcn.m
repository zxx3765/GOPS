function jcob = IFEKF_mea_jcob_fcn(x_k)
dt=0.001;
jcob = [2*cos(x_k(5)*dt),-1,2*sin(x_k(5)*dt),-2*sin(x_k(5)*dt),dt*(-2*x_k(1)*sin(x_k(5)*dt)+2*(x_k(3)-x_k(4))*cos(x_k(5)*dt))];

end