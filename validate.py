
from env.rrpfsp_env import RRPFSPEnv, shop_info_initial
import PPO_model
import torch
import time
import os
import copy
import json


def get_validate_env(env_paras):
    """
    Generate and return the validation environment from the validation set
    """
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    file_path = os.path.abspath('.') + "/data_dev/"

    shop_info = shop_info_initial(load_dict["shop_paras"])
    valid_data_flies = os.listdir(file_path)
    valid_data_flies.sort()
    for i_valid in range(len(valid_data_flies)):
        valid_data_flies[i_valid] = file_path + valid_data_flies[i_valid]

    env_valid = RRPFSPEnv(case=valid_data_flies, shop_info=shop_info, env_paras=env_paras, data_source='file')

    return env_valid


def validate(env_paras, env, model_policy, device):
    """
    Validate the policy during training
    """
    start = time.time()
    batch_size = env_paras["batch_size"]
    env_device = torch.device(env_paras["device"])
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            state.change_device(device)
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        actions = actions.to(env_device)
        state, rewards, dones = env.step(actions)
        done = dones.all()

    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    print('validating time: ', time.time() - start, '\tvalidating mean makespan: ', makespan,
          '\tvalidating makespan: ', makespan_batch, '\n')
    env.reset()

    return makespan, makespan_batch
