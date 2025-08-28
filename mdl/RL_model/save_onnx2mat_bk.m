% Try common sizes step by step
pt_path = 
model_path = 'NN_controller_PPO1.onnx';
mat_path = 'model_onnx.mat';
net = importNetworkFromONNX(model_path);
X = dlarray([0.5,0.5,0.5], 'UU');
layer = inputLayer([1,3],'UU');
net = addInputLayer(net,layer);
net.Initialized
summary(net)
predict(net,X)
save(mat_path,'net')