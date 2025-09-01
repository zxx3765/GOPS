
model_path = 'NN_DDPG_controller84700.onnx';
mat_path = 'NN_DDPG_controller84700.mat';
net = importNetworkFromONNX(model_path);
X = dlarray([0.5,0.5,0.5,0.5], 'UU');
layer = inputLayer([1,4],'UU');
net = addInputLayer(net,layer);
net.Initialized
summary(net)
net = initialize(net, X);
predict(net,X)
save(mat_path,'net')