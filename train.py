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
from validate import get_validate_env, validate
import PPO_model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # PyTorch initialization
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(
        precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    device_model = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    shop_paras = load_dict["shop_paras"]
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] + model_paras["in_size_job"]
    # model_paras["job_selection_dim"] = \
    #     model_paras["out_size_ope"] + model_paras["out_size_ma"] * 2 + model_paras["in_size_job"]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(env_valid_paras)
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
    for i_iter in range(1, train_paras["max_iteration"] + 1):
        print("Iteration: ", i_iter)

        # Replace training instances every x iteration (x = 20 in paper)
        if (i_iter - 1) % train_paras["parallel_iter"] == 0:
            # Generate new training instances and envs
            case = CaseGenerator(shop_info=shop_info, flag_doc=False)
            env = RRPFSPEnv(case=case, shop_info=shop_info, env_paras=env_paras, data_source='case')

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch.to(device_model)
        last_time = time.time()

        # Schedule the parallel instances
        while ~done:
            with torch.no_grad():
                state.change_device(device_model)
                actions = model.policy_old.act(state, memories, dones, flag_sample=True, flag_train=True)
            actions = (actions[0].to(device), actions[1].to(device))
            # print(torch.nonzero(actions[0]))
            # actions = actions.to(device)
            # print("batch_size: ", len(env.batch_idxes))
            state, rewards, dones = env.step(actions)
            # print("feat_opes: ", state.feat_ope_batch)
            # print("rewards: ", rewards)
            done = dones.all()
            memories.rewards.append(copy.deepcopy(rewards).to(device_model))
            memories.is_terminals.append(copy.deepcopy(dones).to(device_model))

        print("Spend time: ", time.time() - last_time)

        # Verify the validity of the schedule
        # gantt_result = env.validate_gantt()[0]
        # if not gantt_result:
        #     print("Scheduling Error！！！！！！")
        env.reset()

        # if iter mod x == 0, then update the policy
        if i_iter % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, train_paras)
            print("reward: ", '%.3f' % reward, "\tloss: ", '%.3f' % loss)
            reward_iter.append(reward)
            loss_iter.append(loss)
            memories.clear_memory()

        # if iter mod x == 0, then validate the policy
        if i_iter % train_paras["save_timestep"] == 0:
            print("\nStart validating")
            # Record the average results and the results on each instance
            valid_result, valid_result_100 = validate(env_valid_paras, env_valid, model.policy_old, device=device_model)
            valid_results.append(valid_result.item())
            valid_results_100.append(valid_result_100)

            # Save the best model
            if valid_result < makespan_best:
                makespan_best = valid_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = "{0}/save_best_{1}.pt".format(save_path, i_iter)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

    # Save the data of training curve to files
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter("{0}/training_ave_{1}.xlsx".format(save_path, str_time))
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    data_file.to_excel(writer_ave, sheet_name="Sheet1", index=False)
    data.to_excel(writer_ave, sheet_name="Sheet1", index=False, startcol=1)
    writer_ave.close()
    # Training curve storage path (value of each validating instance)
    writer_100 = pd.ExcelWriter("{0}/training_100_{1}.xlsx".format(save_path, str_time))
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_100, sheet_name="Sheet1", index=False)
    column = [i_col for i_col in range(100)]
    data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name="Sheet1", index=False, startcol=1)
    writer_100.close()
    # Training curve storage path (reward and loss of each iteration)
    writer_reward_loss = pd.ExcelWriter("{0}/reward_loss_{1}.xlsx".format(save_path, str_time))
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_reward_loss, sheet_name="Sheet1", index=False)
    column = [i_col for i_col in range(2)]
    data = pd.DataFrame(np.array(reward_iter), columns=[1])
    data.to_excel(writer_reward_loss, sheet_name="Sheet1", index=False, startcol=1)
    data = pd.DataFrame(np.array(loss_iter), columns=[2])
    data.to_excel(writer_reward_loss, sheet_name="Sheet1", index=False, startcol=2)
    writer_reward_loss.close()

    print("total_spend_time: ", time.time() - start_time)


if __name__ == "__main__":
    seed = 3019
    setup_seed(seed)
    main()


