import copy
import torch
from itertools import chain


def transpose_list_of_tensors(list_of_tensors):
    max_length = max(len(sublist) for sublist in list_of_tensors)

    transposed_list = [[] for _ in range(max_length)]

    for sublist in list_of_tensors:
        for i_sublist, tensor in enumerate(sublist):
            transposed_list[i_sublist].append(tensor)

    return transposed_list


def memory_flatten(memory, device):
    """
    Flatten the memory
    """

    old_ope_ma_adj = copy.deepcopy(memory.ope_ma_adj[-1]).to(device)
    old_ope_ma_adj_out = copy.deepcopy(memory.ope_ma_adj_out[-1]).to(device)
    old_ope_buf_adj = copy.deepcopy(memory.ope_buf_adj[-1]).to(device)
    old_ope_buf_adj_out = copy.deepcopy(memory.ope_buf_adj_out[-1]).to(device)

    batch_size = memory.batch_idxes[0].size(0)

    old_raw_opes = []
    old_raw_mas = []
    old_raw_buf = []
    old_raw_arc_ma_in = []
    old_raw_arc_ma_out = []
    old_raw_arc_buf_in = []
    old_raw_arc_buf_out = []
    old_raw_job = []
    old_eligible = []
    old_eligible_wait = []
    old_action_envs = []
    old_action_job_envs = []
    memory_rewards = []
    memory_is_terminal = []
    old_logprobs = []
    old_logprobs_job = []

    old_idx = []

    for _ in range(batch_size):
        old_raw_opes.append([])
        old_raw_mas.append([])
        old_raw_buf.append([])
        old_raw_arc_ma_in.append([])
        old_raw_arc_ma_out.append([])
        old_raw_arc_buf_in.append([])
        old_raw_arc_buf_out.append([])
        old_raw_job.append([])
        old_eligible.append([])
        old_eligible_wait.append([])
        old_action_envs.append([])
        old_action_job_envs.append([])
        memory_rewards.append([])
        memory_is_terminal.append([])
        old_logprobs.append([])
        old_logprobs_job.append([])

        old_idx.append([])

    for i_list in range(len(memory.batch_idxes)):
        for i_idx in range(len(memory.batch_idxes[i_list])):
            old_raw_opes[memory.batch_idxes[i_list][i_idx]].append(memory.raw_opes[i_list][i_idx])
            old_raw_mas[memory.batch_idxes[i_list][i_idx]].append(memory.raw_mas[i_list][i_idx])
            old_raw_buf[memory.batch_idxes[i_list][i_idx]].append(memory.raw_buf[i_list][i_idx])
            old_raw_arc_ma_in[memory.batch_idxes[i_list][i_idx]].append(memory.raw_arc_ma_in[i_list][i_idx])
            old_raw_arc_ma_out[memory.batch_idxes[i_list][i_idx]].append(memory.raw_arc_ma_out[i_list][i_idx])
            old_raw_arc_buf_in[memory.batch_idxes[i_list][i_idx]].append(memory.raw_arc_buf_in[i_list][i_idx])
            old_raw_arc_buf_out[memory.batch_idxes[i_list][i_idx]].append(memory.raw_arc_buf_out[i_list][i_idx])
            old_raw_job[memory.batch_idxes[i_list][i_idx]].append(memory.raw_job[i_list][i_idx])
            old_eligible[memory.batch_idxes[i_list][i_idx]].append(memory.eligible[i_list][i_idx])
            old_eligible_wait[memory.batch_idxes[i_list][i_idx]].append(memory.eligible_wait[i_list][i_idx])
            old_action_envs[memory.batch_idxes[i_list][i_idx]].append(memory.action_envs[i_list][i_idx])
            old_action_job_envs[memory.batch_idxes[i_list][i_idx]].append(memory.action_job_envs[i_list][i_idx])
            memory_rewards[memory.batch_idxes[i_list][i_idx]].append(memory.rewards[i_list][memory.batch_idxes[i_list][i_idx]])
            memory_is_terminal[memory.batch_idxes[i_list][i_idx]].append(
                memory.is_terminals[i_list][memory.batch_idxes[i_list][i_idx]])
            old_logprobs[memory.batch_idxes[i_list][i_idx]].append(memory.logprobs[i_list][i_idx])
            old_logprobs_job[memory.batch_idxes[i_list][i_idx]].append(memory.logprobs_job[i_list][i_idx])

            # old_idx[memory.batch_idxes[i_list][i_idx]].append(memory.batch_idxes[i_list][i_idx])


    old_raw_opes = torch.stack(list(chain(*old_raw_opes))).to(device)
    old_raw_mas = torch.stack(list(chain(*old_raw_mas))).to(device)
    old_raw_buf = torch.stack(list(chain(*old_raw_buf))).to(device)
    old_raw_arc_ma_in = torch.stack(list(chain(*old_raw_arc_ma_in))).to(device)
    old_raw_arc_ma_out = torch.stack(list(chain(*old_raw_arc_ma_out))).to(device)
    old_raw_arc_buf_in = torch.stack(list(chain(*old_raw_arc_buf_in))).to(device)
    old_raw_arc_buf_out = torch.stack(list(chain(*old_raw_arc_buf_out))).to(device)
    old_raw_job = list(chain(*old_raw_job))
    old_eligible = torch.stack(list(chain(*old_eligible))).to(device)
    old_eligible_wait = torch.stack(list(chain(*old_eligible_wait))).to(device)
    old_action_envs = torch.stack(list(chain(*old_action_envs))).to(device)
    old_action_job_envs = torch.stack(list(chain(*old_action_job_envs))).to(device)
    memory_rewards = torch.stack(list(chain(*memory_rewards))).to(device)
    memory_is_terminal = torch.stack(list(chain(*memory_is_terminal))).to(device)
    old_logprobs = torch.stack(list(chain(*old_logprobs))).to(device)
    old_logprobs_job = torch.stack(list(chain(*old_logprobs_job))).to(device)

    # old_idx = torch.stack(list(chain(*old_idx))).to(device)

    return old_ope_ma_adj, old_ope_ma_adj_out, old_ope_buf_adj, old_ope_buf_adj_out, old_raw_opes, old_raw_mas, \
        old_raw_buf, old_raw_arc_ma_in, old_raw_arc_ma_out, old_raw_arc_buf_in, old_raw_arc_buf_out, old_raw_job, \
        old_eligible, old_eligible_wait, old_action_envs, old_action_job_envs, memory_rewards, memory_is_terminal, \
        old_logprobs, old_logprobs_job
