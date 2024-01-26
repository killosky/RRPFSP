import json
import os
import random

import torch

from rrpfsp_env import shop_info_initial, RRPFSPEnv


def action_generate(env):
    arc_action_mask = torch.cat((
        env.mask_mas_arc_batch[env.batch_idxes], env.mask_buf_arc_batch[env.batch_idxes]), dim=2)
    arc_action = torch.zeros_like(arc_action_mask, dtype=torch.long)
    job_action = torch.zeros(size=(len(env.batch_idxes),), dtype=torch.long)
    for i_batch_idx in range(len(env.batch_idxes)):
        i_batch_action = env.batch_idxes[i_batch_idx]

        i_action_idx_set = torch.nonzero(arc_action_mask[i_batch_idx, :, :, :])
        if env.mask_wait_batch[i_batch_action]:
            i_action_idx_set = torch.cat((i_action_idx_set, torch.tensor([[0, 0, 2]])), dim=0)
        i_arc_action_selection = random.randint(
            0, i_action_idx_set.size(0) - 1) if i_action_idx_set.size(0) > 0 else []
        arc_action_selection = i_action_idx_set[i_arc_action_selection]
        # print("i_action_idx_set: ", i_action_idx_set)
        # print("arc_action_selection: ", arc_action_selection)
        if arc_action_selection[2] < 2:
            arc_action[i_batch_idx, arc_action_selection[0], arc_action_selection[1], arc_action_selection[2]] = 1
            if arc_action_selection[2] == 0:
                if arc_action_selection[1] < env.station_num:
                    selection_job_set = torch.nonzero(
                        env.ope_node_job_batch[i_batch_action][arc_action_selection[0], :, 0])
                    i_job_selection = random.randint(0, selection_job_set.size(0) - 1)
                    job_action[i_batch_idx] = selection_job_set[i_job_selection]
                else:
                    selection_job_set = torch.nonzero(
                        env.ope_node_job_batch[i_batch_action][arc_action_selection[0], :, 1])
                    i_job_selection = random.randint(0, selection_job_set.size(0) - 1)
                    job_action[i_batch_idx] = selection_job_set[i_job_selection]
            else:
                if arc_action_selection[1] < env.station_num:
                    selection_job_set = torch.nonzero(
                        env.ope_node_job_batch[i_batch_action][arc_action_selection[0], :, 2])
                    i_job_selection = random.randint(0, selection_job_set.size(0) - 1)
                    job_action[i_batch_idx] = selection_job_set[i_job_selection]
                else:
                    selection_job_set = torch.nonzero(
                        env.ope_node_job_batch[i_batch_action][arc_action_selection[0], :, 3])
                    i_job_selection = random.randint(0, selection_job_set.size(0) - 1)
                    job_action[i_batch_idx] = selection_job_set[i_job_selection]

    action = [arc_action, job_action]

    return action


if __name__ == "__main__":
    with open("./shop_info_paras.json", 'r') as load_f:
        load_dict_shop = json.load(load_f)
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    load_dict_env = load_dict["env_paras"]

    path = os.path.abspath('.') + "/data/"

    shop_info = shop_info_initial(load_dict_shop)
    valid_data_flies = os.listdir(path)
    valid_data_flies.sort()
    for i_valid in range(len(valid_data_flies)):
        valid_data_flies[i_valid] = path + valid_data_flies[i_valid]

    env_valid = RRPFSPEnv(case=valid_data_flies, shop_info=shop_info, env_paras=load_dict_env, data_source='file')
    random.seed(1)

    while ~env_valid.done:
        # print("mask_mas: ", torch.nonzero(env_valid.mask_mas_arc_batch))
        # print("mask_buffer: ", torch.nonzero(env_valid.mask_buf_arc_batch))
        # print("mask_wait: ", torch.nonzero(env_valid.mask_wait_batch))
        action = action_generate(env_valid)
        a, b = action
        # if b == 8:
            # print("action: ", torch.nonzero(a), b)
            # print(env_valid.job_to_buf_flag_batch[0][3, :])
            # print("______________________________")
        # print("ope_node_job", torch.nonzero(env_valid.ope_node_job_batch[0]))
        # print(env_valid.job_to_buf_flag_batch)
        print("action: ", torch.nonzero(a), b)
        env_valid.step(action)
        # print(env_valid.done_batch)
        # print(env_valid.batch_idxes)

        # print("__________________________________________________________________________")




