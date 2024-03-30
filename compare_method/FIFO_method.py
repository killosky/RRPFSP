
import torch
import json
import os

from env.rrpfsp_env import RRPFSPEnv, shop_info_initial


def get_test_env(load_dict):
    """
    Generate and return the validation environment from the validation set
    """

    file_path = os.path.abspath('..') + "/data_dev/"

    shop_info = shop_info_initial(load_dict["shop_paras"])
    env_paras = load_dict["env_paras"]
    test_data_flies = os.listdir(file_path)
    test_data_flies.sort()
    for i_valid in range(len(test_data_flies)):
        test_data_flies[i_valid] = file_path + test_data_flies[i_valid]

    env_test = RRPFSPEnv(case=test_data_flies, shop_info=shop_info, env_paras=env_paras, data_source='file')

    return env_test


def test_fifo(env):
    """
    Use FIFO method to solve the scheduling problem.
    """
    arc_action = torch.zeros(size=(len(env.batch_idxes), env.ope_num+1, env.station_num+3, 2), dtype=torch.long)
    job_action = torch.zeros(size=(len(env.batch_idxes),), dtype=torch.long)
    for i_batch_idx in range(len(env.batch_idxes)):

        i_batch_action = env.batch_idxes[i_batch_idx]
        selection_job = None
        selection_arc = None

        idxes_in_mas_arc = torch.nonzero(env.mask_mas_arc_batch[i_batch_action, :, :, 0])
        idxes_out_arc = torch.nonzero(torch.cat(
            (env.mask_mas_arc_batch[i_batch_action, :, :, 1], env.mask_buf_arc_batch[i_batch_action, :, :, 1]), dim=1))
        idxes_in_buf_arc = torch.nonzero(env.mask_buf_arc_batch[i_batch_action, :, :, 0])

        # Consider the operation that will be processed, corresponding to mask_mas_arc_in
        if idxes_in_mas_arc.size(0) > 0:
            for idx_in_mas_arc in idxes_in_mas_arc:
                selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_in_mas_arc[0], :, 0].squeeze(-1) \
                                     * env.mask_job_batch[i_batch_action].long()
                idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                if selection_job is None:
                    selection_job = idx_selection_job
                    selection_arc = idx_in_mas_arc
                else:
                    if selection_job > idx_selection_job:
                        selection_job = idx_selection_job
                        selection_arc = idx_in_mas_arc
            arc_action[i_batch_idx, selection_arc[0], selection_arc[1], 0] = 1
            job_action[i_batch_idx] = selection_job

        # Consider the operation that will be loaded on trans-robot,
        # corresponding to mask_mas_arc_out and mask_buf_arc_out
        elif idxes_out_arc.size(0) > 0:
            for idx_out_arc in idxes_out_arc:
                if idx_out_arc[1] >= env.station_num:
                    selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_out_arc[0], :, 3].squeeze(-1) \
                                         * env.mask_job_batch[i_batch_action].long()
                    idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_out_arc
                    else:
                        if selection_job > idx_selection_job:
                            selection_job = idx_selection_job
                            selection_arc = idx_out_arc

                else:
                    selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_out_arc[0], :, 2].squeeze(-1) \
                                         * env.mask_job_batch[i_batch_action].long()
                    idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_out_arc
                    else:
                        if selection_job > idx_selection_job:
                            selection_job = idx_selection_job
                            selection_arc = idx_out_arc

            arc_action[i_batch_idx, selection_arc[0], selection_arc[1], 1] = 1
            job_action[i_batch_idx] = selection_job

        # Consider the operation that will be transported to buffer, corresponding to mask_buf_arc_in
        elif idxes_in_buf_arc.size(0) > 0:
            for idx_in_buf_arc in idxes_in_buf_arc:
                selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_in_buf_arc[0], :, 1].squeeze(-1) \
                                     * env.mask_job_batch[i_batch_action].long()
                # If the job will be loaded to unloading buffer, then the job will be selected
                if idx_in_buf_arc[1] == 2:
                    idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_in_buf_arc
                    else:
                        if selection_job > idx_selection_job:
                            selection_job = idx_selection_job
                            selection_arc = idx_in_buf_arc
                    break
                else:
                    idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_in_buf_arc
                    else:
                        if selection_job > idx_selection_job:
                            selection_job = idx_selection_job
                            selection_arc = idx_in_buf_arc

            arc_action[i_batch_idx, selection_arc[0], selection_arc[1]+env.station_num, 0] = 1
            job_action[i_batch_idx] = selection_job

    action = [arc_action, job_action]

    return action


def test_spt(env):
    """
    Use SPT method to solve the scheduling problem.
    """
    arc_action = torch.zeros(size=(len(env.batch_idxes), env.ope_num+1, env.station_num+3, 2), dtype=torch.long)
    job_action = torch.zeros(size=(len(env.batch_idxes),), dtype=torch.long)
    for i_batch_idx in range(len(env.batch_idxes)):

        i_batch_action = env.batch_idxes[i_batch_idx]
        selection_job = None
        selection_arc = None
        processing_time_max = None

        idxes_in_mas_arc = torch.nonzero(env.mask_mas_arc_batch[i_batch_action, :, :, 0])
        idxes_out_arc = torch.nonzero(torch.cat(
            (env.mask_mas_arc_batch[i_batch_action, :, :, 1], env.mask_buf_arc_batch[i_batch_action, :, :, 1]), dim=1))
        idxes_in_buf_arc = torch.nonzero(env.mask_buf_arc_batch[i_batch_action, :, :, 0])

        # Consider the operation that will be processed, corresponding to mask_mas_arc_in
        if idxes_in_mas_arc.size(0) > 0:
            for idx_in_mas_arc in idxes_in_mas_arc:
                selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_in_mas_arc[0], :, 0].squeeze(-1) \
                                     * env.mask_job_batch[i_batch_action].long()
                idx_selection_job_set = torch.nonzero(selection_job_mask).squeeze(-1)
                processing_time_set = env_test.proc_time_batch[i_batch_action][
                    idx_selection_job_set, env_test.job_next_ope[i_batch_action][idx_selection_job_set]]
                idx_selection_job = idx_selection_job_set[torch.argmin(processing_time_set)]
                processing_time = torch.min(processing_time_set)
                if selection_job is None:
                    selection_job = idx_selection_job
                    selection_arc = idx_in_mas_arc
                    processing_time_max = processing_time
                else:
                    if processing_time_max < processing_time:
                        selection_job = idx_selection_job
                        selection_arc = idx_in_mas_arc
                        processing_time_max = processing_time
            arc_action[i_batch_idx, selection_arc[0], selection_arc[1], 0] = 1
            job_action[i_batch_idx] = selection_job

        # Consider the operation that will be loaded on trans-robot,
        # corresponding to mask_mas_arc_out and mask_buf_arc_out
        elif idxes_out_arc.size(0) > 0:
            for idx_out_arc in idxes_out_arc:
                if idx_out_arc[1] >= env.station_num:
                    selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_out_arc[0], :, 3].squeeze(-1) \
                                         * env.mask_job_batch[i_batch_action].long()
                    idx_selection_job_set = torch.nonzero(selection_job_mask).squeeze(-1)
                    processing_time_set = env_test.proc_time_batch[i_batch_action][
                        idx_selection_job_set, env_test.job_next_ope[i_batch_action][idx_selection_job_set]]
                    idx_selection_job = idx_selection_job_set[torch.argmin(processing_time_set)]
                    processing_time = torch.min(processing_time_set)
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_out_arc
                        processing_time_max = processing_time
                    else:
                        if processing_time_max < processing_time:
                            selection_job = idx_selection_job
                            selection_arc = idx_out_arc
                            processing_time_max = processing_time

                else:
                    selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_out_arc[0], :, 2].squeeze(-1) \
                                         * env.mask_job_batch[i_batch_action].long()
                    idx_selection_job_set = torch.nonzero(selection_job_mask).squeeze(-1)
                    processing_time_set = env_test.proc_time_batch[i_batch_action][
                        idx_selection_job_set, env_test.job_next_ope[i_batch_action][idx_selection_job_set]]

                    idx_selection_job = idx_selection_job_set[torch.argmin(processing_time_set)]
                    processing_time = torch.min(processing_time_set)
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_out_arc
                        processing_time_max = processing_time
                    else:
                        if processing_time_max < processing_time:
                            selection_job = idx_selection_job
                            selection_arc = idx_out_arc
                            processing_time_max = processing_time

            arc_action[i_batch_idx, selection_arc[0], selection_arc[1], 1] = 1
            job_action[i_batch_idx] = selection_job

        # Consider the operation that will be transported to buffer, corresponding to mask_buf_arc_in
        elif idxes_in_buf_arc.size(0) > 0:
            for idx_in_buf_arc in idxes_in_buf_arc:
                selection_job_mask = env.ope_node_job_batch[i_batch_action][idx_in_buf_arc[0], :, 1].squeeze(-1) \
                                     * env.mask_job_batch[i_batch_action].long()
                # If the job will be loaded to unloading buffer, then the job will be selected
                if idx_in_buf_arc[1] == 2:
                    idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_in_buf_arc
                    else:
                        if selection_job > idx_selection_job:
                            selection_job = idx_selection_job
                            selection_arc = idx_in_buf_arc
                    break
                else:
                    idx_selection_job = torch.min(torch.nonzero(selection_job_mask))
                    if selection_job is None:
                        selection_job = idx_selection_job
                        selection_arc = idx_in_buf_arc
                    else:
                        if selection_job > idx_selection_job:
                            selection_job = idx_selection_job
                            selection_arc = idx_in_buf_arc

            arc_action[i_batch_idx, selection_arc[0], selection_arc[1] + env.station_num, 0] = 1
            job_action[i_batch_idx] = selection_job

        action = [arc_action, job_action]

        return action


if __name__ == "__main__":

    with open("../config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    load_dict["env_paras"]["batch_size"] = load_dict["env_paras"]["test_batch_size"]

    env_test = get_test_env(load_dict)
    while not env_test.done:
        action = test_spt(env_test)
        env_test.step(action)

    print("The makespan of FIFO method is: ", env_test.makespan_batch, env_test.makespan_batch.mean())















