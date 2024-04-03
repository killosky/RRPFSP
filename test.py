
import time
import copy
import json
import os

import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model

from env.rrpfsp_env import RRPFSPEnv, shop_info_initial


def get_test_env(env_paras):
    """
    Generate and return the test environment from the validation set
    """
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    file_path = os.path.abspath('.') + "/data_test/"

    shop_info = shop_info_initial(load_dict["shop_paras"])
    valid_data_flies = os.listdir(file_path)
    valid_data_flies.sort()
    for i_valid in range(len(valid_data_flies)):
        valid_data_flies[i_valid] = file_path + valid_data_flies[i_valid]

    env_test = RRPFSPEnv(case=valid_data_flies, shop_info=shop_info, env_paras=env_paras, data_source='file')

    return env_test


def test():
    # PyTorch initialization
    pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cpu")
    device_model = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(
        precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    with open('./config.json', 'r') as load_f:
        load_dict = json.load(load_f)

    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_test_paras = copy.deepcopy(env_paras)
    env_test_paras["batch_size"] = env_paras["test_batch_size"]

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] + model_paras[
        "in_size_job"]

    env_test = get_test_env(env_test_paras)

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)

    mod_files = os.listdir('./model/')[:]
    model_CKPT = torch.load('./model/{0}'.format(mod_files[0]))
    model.policy.load_state_dict(model_CKPT)
    model.policy_old.load_state_dict(model_CKPT)

    makespan_batch = schedule(env_test, model, memories, device, device_model, flag_sample=False)
    writer_ave = pd.ExcelWriter("test_result_DRL.xlsx")
    data = pd.DataFrame(np.array(makespan_batch).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name="Sheet1", index=False, startcol=1)
    writer_ave.close()

    return makespan_batch, makespan_batch.mean()


def schedule(env_test, model, memories, device, device_model, flag_sample=False):
    state = env_test.state
    dones = env_test.done_batch.to(device_model)
    done = False
    run_time = []
    while ~done:
        state.change_device(device_model)
        time_start = time.time()
        with torch.no_grad():
            state.change_device(device_model)
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        run_time.append(time.time() - time_start)
        actions = (actions[0].to(device), actions[1].to(device))
        state, reward, dones = env_test.step(actions)
        done = dones.all()

    # gantt_result = env_test.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error！！！！！！")
    run_time_average = np.array(run_time).mean()
    run_time_total = np.array(run_time).sum()
    print("average step run time:", run_time_average, "\nTotal run time:", run_time_total)

    return copy.deepcopy(env_test.makespan_batch)


if __name__ == "__main__":
    print(test())

