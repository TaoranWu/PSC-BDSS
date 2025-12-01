import numpy as np
from src.utils.sample_disturbance import sample_disturbance

def sim_traj_stochastic(model, init_state, sim_time):
    traj_all = []
    for i in range(len(init_state)):
        traj = [init_state[i]]
        x = np.array(init_state[i]).reshape(1, -1)
        for j in range(sim_time[i]):
            d = sample_disturbance(1, model.disturbance_type, model.d_para)
            x = model.fx(x, d)
            x = np.array(x)
            x = x.reshape(2)
            traj.append(x)
            x = x.reshape(1, -1)
        traj_all.append(traj)
    return traj_all







