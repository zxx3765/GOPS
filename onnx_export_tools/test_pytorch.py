import numpy as np, torch, torch.jit
def model_inference(model, obs):
    obs = torch.from_numpy(obs)
    with torch.no_grad():
        act = model(obs)
    return act.numpy()
import time

model = torch.jit.load(r'D:\Project\SynologyDrive\Imp_fcn_cal\RL_model\NN_controller_PPO.pt')
obs = np.array([[0.5, 0.5, 0.5]],dtype=np.float32)
time0 = time.time()
for i in range(100):
    act = model_inference(model, obs)
time1 = time.time()
print("time:", time1-time0)
print(act)
print(obs.shape)
print("end")
    
