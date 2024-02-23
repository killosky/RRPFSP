import copy
import json
import os
import random
import time
from collections import deque

import pandas as pd
import torch
import numpy as np

from env.case_generate import CaseGenerator
from env.rrpfsp_env import RRPFSPEnv, shop_info_initial
import PPO_model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # PyTorch initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(
        precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    shop_paras = load_dict["shop_paras"]
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras["job_selection_dim"] = \
        model_paras["out_size_ope"] + model_paras["out_size_ma"] * 2 + model_paras["in_size_job"]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    # env_valid = get_validate_env(env_valid_paras)
    maxlen = 1    # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    save_path = "./save/train_{0}".format(str_time)
    os.makedirs(save_path)

    valid_results = []
    valid_results_100 = []
    reward_iter = []
    loss_iter = []

    # Confirm the shop information
    shop_info = shop_info_initial(shop_paras)

    # Start training iteration
    start_time = time.time()
    env = None
    for i_iter in range(1, train_paras["max_iter"] + 1):
        print("Iteration: ", i_iter)

        # Replace training instances every x iteration (x = 20 in paper)
        if (i_iter - 1) % train_paras["parallel_iter"] == 0:
            # Generate new training instances and envs
            case = CaseGenerator(shop_info=shop_info, flag_doc=False)
            env = RRPFSPEnv(case=case, shop_info=shop_info, env_paras=env_paras, data_source='case')

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch.to(device)
        last_time = time.time()

        # Schedule the parallel instances
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones, flag_sample=False, flag_train=False)
                actions = actions.to(device)
                state, rewards, dones = env.step(actions)
                done = dones.all()
                memories.rewards.append(copy.deepcopy(rewards))
                memories.is_terminals.append(copy.deepcopy(dones))

        print("Spend time: ", time.time() - last_time)

        # Verify the validity of the schedule
        # gantt_result = env.validate_gantt()[0]
        # if not gantt_result:
        #     print("Scheduling Error！！！！！！")
        env.reset()





