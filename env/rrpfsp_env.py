import gym
import torch
import json
import copy
import os
import random

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from env.case_generate import ShopInfo, CaseGenerator, trans_time_generate, job_num_detec
from env.load_data import load_rrpfsp


@dataclass
class EnvState:
    """
    Class for the state of the environment
    """
    # static
    proc_time_batch: list[torch.Tensor] = None
    job_num_batch: torch.Tensor = None
    trans_time: torch.Tensor = None
    load_time: torch.Tensor = None
    unload_time: torch.Tensor = None
    paral_mas_num: torch.Tensor = None
    station_num: torch.Tensor = None
    ope_num: torch.Tensor = None
    routing: torch.Tensor = None
    ope_ma_adj: torch.Tensor = None
    ope_ma_adj_out: torch.Tensor = None
    ope_buf_adj: torch.Tensor = None
    ope_buf_adj_out: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    mas_state_batch: torch.Tensor = None
    mas_left_proctime_batch: torch.Tensor = None

    feat_ope_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    feat_buf_batch: torch.Tensor = None

    feat_arc_ma_in_batch: torch.Tensor = None
    feat_arc_ma_out_batch: torch.Tensor = None
    feat_arc_buf_in_batch: torch.Tensor = None
    feat_arc_buf_out_batch: torch.Tensor = None

    feat_job_batch: list[torch.Tensor] = None
    job_next_ope: list[torch.Tensor] = None
    ope_job_batch: list[torch.Tensor] = None

    job_loc_batch: list[torch.Tensor] = None
    job_loc_ma_batch: list[torch.Tensor] = None
    job_state_batch: list[torch.Tensor] = None
    done_job_batch: list[torch.Tensor] = None
    ope_node_job_batch: list[torch.Tensor] = None
    num_jobs_batch: torch.Tensor = None
    robot_loc_batch: torch.Tensor = None

    mask_mas_arc_batch: torch.Tensor = None
    mask_buf_arc_batch: torch.Tensor = None
    mask_job_batch: list[torch.Tensor] = None
    mask_wait_batch: torch.Tensor = None

    def update(self, batch_idxes, mas_state_batch, mas_left_proctime_batch, feat_ope_batch, feat_mas_batch,
               feat_buf_batch, feat_arc_ma_in_batch, feat_arc_ma_out_batch, feat_arc_buf_in_batch,
               feat_arc_buf_out_batch, feat_job_batch, job_next_ope, ope_job_batch, job_loc_batch, job_loc_ma_batch,
               job_state_batch, done_job_batch, ope_node_job_batch, num_jobs_batch, robot_loc_batch, mask_mas_arc_batch,
               mask_buf_arc_batch, mask_job_batch, mask_wait_batch):
        self.batch_idxes = batch_idxes
        self.mas_state_batch = mas_state_batch
        self.mas_left_proctime_batch = mas_left_proctime_batch
        self.feat_ope_batch = feat_ope_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_buf_batch = feat_buf_batch
        self.feat_arc_ma_in_batch = feat_arc_ma_in_batch
        self.feat_arc_ma_out_batch = feat_arc_ma_out_batch
        self.feat_arc_buf_in_batch = feat_arc_buf_in_batch
        self.feat_arc_buf_out_batch = feat_arc_buf_out_batch
        self.feat_job_batch = feat_job_batch
        self.job_next_ope = job_next_ope
        self.ope_job_batch = ope_job_batch
        self.job_loc_batch = job_loc_batch
        self.job_loc_ma_batch = job_loc_ma_batch
        self.ope_node_job_batch = ope_node_job_batch

        self.num_jobs_batch = num_jobs_batch
        self.robot_loc_batch = robot_loc_batch
        self.job_state_batch = job_state_batch
        self.done_job_batch = done_job_batch
        self.mask_mas_arc_batch = mask_mas_arc_batch
        self.mask_buf_arc_batch = mask_buf_arc_batch
        self.mask_job_batch = mask_job_batch
        self.mask_wait_batch = mask_wait_batch


def shop_info_initial(shop_paras, trans_time_matrix=None):
    if trans_time_matrix is None:
        trans_time_matrix = trans_time_generate(shop_paras["num_stations"], shop_paras["trans_time_ave"],
                                                shop_paras["trans_time_dev"])
    shop_info = ShopInfo(job_min=shop_paras["job_min"], job_max=shop_paras["job_max"],
                         paral_mas_num=shop_paras["paral_mas_num"], routing=shop_paras["routing"],
                         proc_time_min=shop_paras["proc_time_min"], proc_time_max=shop_paras["proc_time_max"],
                         trans_time=trans_time_matrix, trans_cap=shop_paras["trans_cap"],
                         buffer_cap=shop_paras["buffer_cap"], load_time=shop_paras["load_time"],
                         unload_time=shop_paras["unload_time"], proc_time_dev_para=shop_paras["proc_time_dev_para"])
    return shop_info


class RRPFSPEnv(gym.Env):
    """
    RRPFSP environment
    """

    def __init__(self, case, shop_info, env_paras, data_source='case'):
        """
        :param case: RRPFSP instance generator or the addresses of the instances
        Initialize the RRPFSP simulation environment
        """
        # load paras
        # static
        self.device = torch.device(env_paras["device"])
        self.batch_size = env_paras["batch_size"]
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)

        # load instance
        lines = []
        nums_jobs = []
        if data_source == 'case':    # Generate instances through case generator
            for i_batch_case in range(self.batch_size):
                # Generate an instance and save it
                c = CaseGenerator(shop_info, flag_doc=False)
                instance, num_jobs = c.get_case()
                lines.append(instance)
                nums_jobs.append(num_jobs)
        else:    # Load instances from files
            for i_batch_case in range(self.batch_size):
                # Load the instance from the file
                with open(case[i_batch_case], 'r') as f:
                    lines.append(f.readlines())
                num_jobs = job_num_detec(lines[i_batch_case])
                nums_jobs.append(num_jobs)

        ### ___________________________ ###

        # load data
        self.num_jobs_batch = torch.tensor(nums_jobs, dtype=torch.long, device=self.device)
        self.paral_mas_num = torch.tensor(shop_info.paral_mas_num, dtype=torch.long, device=self.device)
        self.station_num = torch.tensor(len(self.paral_mas_num)).long().to(self.device)
        self.mas_num = torch.sum(self.paral_mas_num).long().to(self.device)
        self.ope_num = torch.tensor(shop_info.ope_num).long().to(self.device)
        self.routing = torch.tensor(shop_info.routing).long().to(self.device)
        self.trans_time = torch.tensor(shop_info.trans_time, dtype=torch.float, device=self.device)
        self.load_time = torch.tensor(shop_info.load_time, dtype=torch.float, device=self.device)
        self.unload_time = torch.tensor(shop_info.unload_time).float().to(self.device)
        self.trans_cap = torch.tensor(shop_info.trans_cap, dtype=torch.long, device=self.device)
        self.buffer_cap = torch.tensor(shop_info.buffer_cap, dtype=torch.long, device=self.device)

        self.ope_ma_adj = torch.zeros(size=(self.ope_num+1, self.station_num), device=self.device)
        self.ope_ma_adj_out = torch.zeros(size=(self.ope_num+1, self.station_num), device=self.device)
        for i in range(self.ope_num):
            self.ope_ma_adj[i][self.routing[i]-1] = 1
            self.ope_ma_adj_out[i+1][self.routing[i]-1] = 1

        self.ope_buf_adj = torch.zeros(size=(self.ope_num+1, 3), device=self.device)
        self.ope_buf_adj[self.ope_num][2] = 1
        self.ope_buf_adj[:self.ope_num, 1] = 1
        self.ope_buf_adj_out = torch.zeros(size=(self.ope_num+1, 3), device=self.device)
        self.ope_buf_adj_out[0][0] = 1
        self.ope_buf_adj_out[:self.ope_num, 1] = 1

        # machines feature
        self.mas_state_batch = torch.zeros(size=(self.batch_size, self.mas_num), device=self.device, dtype=torch.long)
        self.mas_left_proctime_batch = torch.zeros(
            size=(self.batch_size, self.mas_num), device=self.device, dtype=torch.float)
        self.mas_appertain_batch = torch.cumsum(self.paral_mas_num, dim=0) - self.paral_mas_num

        self.proc_time_batch = []
        self.left_proc_time_batch = []
        self.job_loc_batch = []
        self.job_loc_ma_batch = []
        self.mask_job_batch = []
        self.job_next_ope = []
        self.job_state_batch = []
        self.ope_job_batch = []
        self.done_job_batch = []
        self.ope_node_job_batch = []
        self.job_to_buf_flag_batch = []

        for i_batch_case in range(self.batch_size):
            proc_time_matrix = load_rrpfsp(lines[i_batch_case], shop_info, device=self.device)
            self.proc_time_batch.append(proc_time_matrix)

            self.left_proc_time_batch.append(proc_time_matrix)

            job_loc_matrix = torch.zeros(
                size=(self.num_jobs_batch[i_batch_case], self.station_num+3), device=self.device)
            job_loc_matrix[:, self.station_num] = 1
            self.job_loc_batch.append(job_loc_matrix)

            job_loc_ma_matrix = torch.zeros(
                size=(self.num_jobs_batch[i_batch_case], self.mas_num+3), device=self.device)
            job_loc_ma_matrix[:, self.mas_num] = 1
            self.job_loc_ma_batch.append(job_loc_ma_matrix)

            mask_job = torch.full(
                size=(self.num_jobs_batch[i_batch_case],), fill_value=True, device=self.device, dtype=torch.bool)
            self.mask_job_batch.append(mask_job)

            self.job_next_ope.append(
                torch.zeros(size=(self.num_jobs_batch[i_batch_case],), device=self.device, dtype=torch.long))

            self.job_state_batch.append(
                torch.zeros(size=(self.num_jobs_batch[i_batch_case],), device=self.device, dtype=torch.long))

            ope_job = torch.zeros(
                size=(self.ope_num+1, self.num_jobs_batch[i_batch_case]), device=self.device, dtype=torch.bool)
            ope_job[0, :] = 1
            self.ope_job_batch.append(ope_job)

            done_job = torch.zeros(
                size=(self.num_jobs_batch[i_batch_case],), device=self.device, dtype=torch.bool)
            self.done_job_batch.append(done_job)

            # size = (self.ope_num+1, self.num_jobs_batch[i_batch_case], 4),
            # corresponding to [mas_in, buf_in, mas_out, buf_out]
            ope_node_job = torch.zeros(
                size=(self.ope_num+1, self.num_jobs_batch[i_batch_case], 4), device=self.device, dtype=torch.long)
            ope_node_job[0, :, 3] = 1
            self.ope_node_job_batch.append(ope_node_job)

            job_to_buf_flag = torch.ones(
                size=(self.num_jobs_batch[i_batch_case], self.ope_num+1), device=self.device, dtype=torch.bool)
            job_to_buf_flag[:, 0] = job_to_buf_flag[:, -1] = False
            self.job_to_buf_flag_batch.append(job_to_buf_flag)

        self.robot_loc_batch = torch.zeros(size=(self.batch_size, self.station_num+3), device=self.device)
        self.robot_loc_batch[:, self.station_num] = 1

        self.mask_mas_arc_batch = torch.zeros(
            size=(self.batch_size, self.ope_num+1, self.station_num, 2), device=self.device, dtype=torch.bool)
        self.mask_buf_arc_batch = torch.zeros(
            size=(self.batch_size, self.ope_num+1, 3, 2), device=self.device, dtype=torch.bool)
        self.mask_buf_arc_batch[:, 0, 0, 1] = True

        self.mask_wait_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.bool)

        '''
        features, dynamic
            opes:
                1. number of jobs waiting for processing the operation 
                (The job which has finished processing last operation, 
                and this ope has not been processed)
                2. number of the corresponding jobs on the machine
                3. number of the corresponding jobs waiting in the buffer
                4. the completion rate of this ope
                5. left time of the operation being processed on the machine
                6. sum of the processing time of the waiting operations
                7. minimum of the left processing time for the job being processed 
            mas:
                1. number of idle machines in the set
                2. number of finished machines
                3. The non-zero minimum remaining processing time of the on-going machine
                4. number of jobs waiting and being processed on the machine
                5. sum of left processing time of the job waiting and being processed on the machine
                6. Trans-robot location
            buffer:
                1. number of jobs waiting in the buffer
                2. number of ope nodes in the neighborhood of the buffer
                3. sum of processing time of the next operation of the jobs waiting in the buffer
                4. sum of left processing time of the jobs waiting in the buffer
                5. average state of the machine required for the next step of the job on it
                6. Trans-robot location
            arc:
                1. processing time for the operation of the arc
                2. transportation time for the arc
                3. number of jobs on the trans robot
            jobs:
                1. the left processing time for the operation being processed or will be processed for the job
                2. the total left processing time for the job
                3. number of left operations of the job
        '''
        # Generate raw feature vectors of operations
        feat_ope_batch = torch.zeros(size=(self.batch_size, self.ope_num+1, 7), device=self.device, dtype=torch.float)
        feat_ope_batch[:, 0, 0] = self.num_jobs_batch
        for i_batch_case in range(self.batch_size):
            feat_ope_batch[i_batch_case, :self.ope_num, 4] = torch.sum(
                self.proc_time_batch[i_batch_case][:, :self.ope_num], dim=0)
            feat_ope_batch[i_batch_case, 0, 6] = torch.min(
                self.proc_time_batch[i_batch_case][:, 0])
        feat_ope_batch[:, 0, 5] = feat_ope_batch[:, 0, 4]

        self.feat_ope_batch = feat_ope_batch

        # Generate raw feature vectors of machines
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.station_num, 6), device=self.device, dtype=torch.float)
        feat_mas_batch[:, :, 0] = self.paral_mas_num.unsqueeze(0).expand(self.batch_size, -1)
        feat_mas_batch[:, self.routing[0]-1, 3] = self.num_jobs_batch
        feat_mas_batch[:, self.routing[0]-1, 4] = feat_ope_batch[:, 0, 4]

        self.feat_mas_batch = feat_mas_batch

        # Generate raw feature vectors of buffers
        feat_buf_batch = torch.zeros(size=(self.batch_size, 3, 6), device=self.device, dtype=torch.float)
        feat_buf_batch[:, 0, 0] = self.num_jobs_batch

        feat_buf_batch[:, 0, 1] = feat_buf_batch[:, 2, 1] = 1
        feat_buf_batch[:, 1, 1] = self.ope_num * 2

        feat_buf_batch[:, 0, 2] = feat_ope_batch[:, 0, 4]
        feat_buf_batch[:, 0, 3] = torch.sum(feat_ope_batch[:, :, 4], dim=1)
        feat_buf_batch[:, 0, 5] = 1

        self.feat_buf_batch = feat_buf_batch

        # Generate raw feature vectors of arcs
        self.feat_arc_ma_in_batch = torch.zeros(size=(self.batch_size, self.ope_num+1, self.station_num, 3),
                                                device=self.device, dtype=torch.float)
        self.feat_arc_ma_out_batch = torch.zeros_like(self.feat_arc_ma_in_batch)
        self.feat_arc_buf_in_batch = torch.zeros(size=(self.batch_size, self.ope_num+1, 3, 3),
                                                 device=self.device, dtype=torch.float)
        self.feat_arc_buf_out_batch = torch.zeros_like(self.feat_arc_buf_in_batch)
        self.feat_arc_buf_out_batch[:, 0, 0, 0] = feat_ope_batch[:, 0, 4] / feat_ope_batch[:, 0, 0]
        self.feat_arc_buf_out_batch[:, 0, 0, 1] = self.load_time

        # Generate raw feature vectors of jobs
        feat_job_batch = []
        for i_batch_case in range(self.batch_size):
            feat_job = torch.zeros(size=(self.num_jobs_batch[i_batch_case], 3), device=self.device, dtype=torch.float)
            feat_job[:, 0] = self.proc_time_batch[i_batch_case][:, 0]
            feat_job[:, 1] = torch.sum(self.proc_time_batch[i_batch_case], dim=1)
            feat_job[:, 2] = self.ope_num
            feat_job_batch.append(feat_job)
        self.feat_job_batch = feat_job_batch

        """
        Schedule Information of the environment
            job idxes
            start processing time
            end processing time
            end occupied time
        """
        schedule_batch = []
        for i_batch_case in range(self.batch_size):
            schedule = []
            for _ in range(self.mas_num+3):
                schedule.append([])
            schedule_batch.append(schedule)
        self.schedule_batch = schedule_batch

        self.time = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)
        self.makespan_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)
        self.done_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.bool)
        self.done = self.done_batch.all()
        self.reward_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)
        self.cum_reward_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)

        self.state = EnvState(proc_time_batch=self.proc_time_batch, job_num_batch=self.num_jobs_batch,
                              trans_time=self.trans_time, load_time=self.load_time, unload_time=self.unload_time,
                              paral_mas_num=self.paral_mas_num, station_num=self.station_num, ope_num=self.ope_num,
                              routing=self.routing, ope_ma_adj=self.ope_ma_adj, ope_ma_adj_out=self.ope_ma_adj_out,
                              ope_buf_adj=self.ope_buf_adj, ope_buf_adj_out=self.ope_buf_adj_out,
                              batch_idxes=self.batch_idxes, mas_state_batch=self.mas_state_batch,
                              mas_left_proctime_batch=self.mas_left_proctime_batch, feat_ope_batch=self.feat_ope_batch,
                              feat_mas_batch=self.feat_mas_batch, feat_buf_batch=self.feat_buf_batch,
                              feat_arc_ma_in_batch=self.feat_arc_ma_in_batch,
                              feat_arc_ma_out_batch=self.feat_arc_ma_out_batch,
                              feat_arc_buf_in_batch=self.feat_arc_buf_in_batch,
                              feat_arc_buf_out_batch=self.feat_arc_buf_out_batch,
                              feat_job_batch=self.feat_job_batch, job_next_ope=self.job_next_ope,
                              ope_job_batch=self.ope_job_batch, job_loc_batch=self.job_loc_batch,
                              job_loc_ma_batch=self.job_loc_ma_batch, done_job_batch=self.done_job_batch,
                              ope_node_job_batch=self.ope_node_job_batch, job_state_batch=self.job_state_batch,
                              num_jobs_batch=self.num_jobs_batch, robot_loc_batch=self.robot_loc_batch,
                              mask_mas_arc_batch=self.mask_mas_arc_batch, mask_buf_arc_batch=self.mask_buf_arc_batch,
                              mask_job_batch=self.mask_job_batch, mask_wait_batch=self.mask_wait_batch)

        # Save the initial data for reset
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):
        """
        Take actions and return the next state, reward, and done.
        Action shape: [arc matrix with shape (len(batch idx), num_opes+1, num_stations+3, 2), job idx]
        """
        arc_actions, job_actions = actions

        self.reward_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)

        add_time_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)

        # update the state information of the environment directly influenced by the actions
        for i_idxes in range(len(self.batch_idxes)):
            i_batch = self.batch_idxes[i_idxes]
            if self.done_batch[i_batch]:
                continue
            else:
                wait_flag = ~torch.sum(arc_actions[i_idxes], dim=(-1, -2, -3)).bool()

                if wait_flag:
                    positive_next_time = self.feat_ope_batch[i_batch, :, 6][self.feat_ope_batch[i_batch, :, 6] > 0.]
                    add_time_batch[i_batch] = torch.min(
                        positive_next_time) if positive_next_time.nelement() > 0 else torch.tensor(0)

                if ~wait_flag:
                    # get the idxes of the action
                    action_idx = torch.nonzero(arc_actions[i_idxes, :])
                    action_ope, action_station, out_flag = action_idx[0, 0], action_idx[0, 1], action_idx[0, 2]
                    out_flag = out_flag.bool()
                    in_flag = ~out_flag
                    # action_job = torch.nonzero(
                    #     self.ope_job_batch[i_batch][action_ope, :])[job_actions[i_batch]].squeeze()
                    action_job = job_actions[i_idxes]

                    # Initial the location of the robot
                    self.robot_loc_batch[i_batch] = 0
                    self.feat_mas_batch[i_batch, :, 5] = 0
                    self.feat_buf_batch[i_batch, :, 5] = 0

                    if in_flag:
                        if action_station < self.station_num:
                            # select machine in the station
                            mas_idx = self.mas_appertain_batch[action_station] + torch.nonzero(self.mas_state_batch[
                                        i_batch, self.mas_appertain_batch[action_station]:self.mas_appertain_batch[
                                            action_station]+self.paral_mas_num[action_station]] == 0).squeeze(-1)[0]
                            # update mas state influenced by the actions directly
                            self.mas_state_batch[i_batch, mas_idx] = 1
                            self.mas_left_proctime_batch[i_batch, mas_idx] = self.proc_time_batch[
                                    i_batch][action_job][action_ope] + self.feat_arc_ma_in_batch[
                                        i_batch, action_ope, action_station, 1]
                            self.left_proc_time_batch[i_batch][action_job][action_ope] += self.feat_arc_ma_in_batch[
                                        i_batch, action_ope, action_station, 1]
                            self.feat_mas_batch[i_batch, action_station, 0] -= 1
                            self.feat_mas_batch[i_batch, action_station, 5] = 1

                            self.job_loc_batch[i_batch][job_actions[i_idxes], action_station] = 1
                            self.job_loc_ma_batch[i_batch][job_actions[i_idxes], mas_idx] = 1
                            self.job_state_batch[i_batch][job_actions[i_idxes]] = 1

                            self.feat_ope_batch[i_batch, action_ope, 0] -= 1

                            start_proc_time = self.time[i_batch] + self.feat_arc_ma_in_batch[
                                i_batch, action_ope, action_station, 1]
                            self.schedule_batch[i_batch][mas_idx].append([action_job.item(), start_proc_time.item(),
                                                                          start_proc_time.item(),
                                                                          start_proc_time.item()])
                            # print("in_action", self.schedule_batch[i_batch])

                            add_time_batch[i_batch] = self.feat_arc_ma_in_batch[i_batch, action_ope, action_station, 1]
                            self.reward_batch[i_batch] += self.proc_time_batch[i_batch][action_job][action_ope]

                        else:
                            self.job_loc_batch[i_batch][job_actions[i_idxes], action_station] = 1
                            self.job_loc_ma_batch[i_batch][job_actions[i_idxes], action_station-self.station_num-3] = 1
                            if action_station == self.station_num + 2:
                                self.ope_job_batch[i_batch][action_ope, action_job] = 0
                                self.done_job_batch[i_batch][action_job] = True
                                self.done_batch[i_batch] = self.done_job_batch[i_batch].all()
                                self.makespan_batch[i_batch] = self.time[i_batch] + self.feat_arc_buf_in_batch[
                                    i_batch, action_ope, action_station - self.station_num - 3, 1]
                            elif action_station == self.station_num + 1:
                                self.feat_ope_batch[i_batch, action_ope, 2] -= 1
                                self.job_to_buf_flag_batch[i_batch][action_job, action_ope] = False
                            else:
                                pass
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 0] -= 1
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 2] += self.proc_time_batch[
                                i_batch][action_job][self.job_next_ope[i_batch][action_job]]
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 3] += torch.sum(
                                self.proc_time_batch[i_batch][action_job, self.job_next_ope[i_batch][action_job]:])
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 4] = self.feat_buf_batch[
                                i_batch, action_station-self.station_num-3, 3] / torch.sum(torch.sum(
                                    self.proc_time_batch[i_batch][action_job, :]))

                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 5] = 1
                            start_proc_time = self.time[i_batch] + self.feat_arc_buf_in_batch[
                                i_batch, action_ope, action_station-self.station_num-3, 1]
                            self.schedule_batch[i_batch][action_station-self.station_num-3].append(
                                [action_job.item(), start_proc_time.item(),
                                 start_proc_time.item(), start_proc_time.item()])
                            add_time_batch[i_batch] = self.feat_arc_buf_in_batch[
                                i_batch, action_ope, action_station - self.station_num - 3, 1]

                        self.robot_loc_batch[i_batch, action_station] = 1
                        self.feat_ope_batch[i_batch, action_ope, 1] -= 1

                        self.feat_arc_ma_in_batch[i_batch, :, :, 2] -= 1
                        self.feat_arc_ma_out_batch[i_batch, :, :, 2] -= 1
                        self.feat_arc_buf_in_batch[i_batch, :, :, 2] -= 1
                        self.feat_arc_buf_out_batch[i_batch, :, :, 2] -= 1

                    else:
                        if action_station < self.station_num:
                            # get the id of the machine
                            mas_idx = torch.nonzero(
                                self.job_loc_ma_batch[i_batch][job_actions[i_idxes], :]).squeeze(-1)[0]
                            # update mas state influenced by the actions directly
                            self.mas_state_batch[i_batch, mas_idx] = 0
                            self.feat_mas_batch[i_batch, action_station, 0] += 1
                            self.feat_mas_batch[i_batch, action_station, 1] -= 1

                            # update the job state influenced by the actions directly
                            self.job_state_batch[i_batch][action_job] = 0

                            # update feat_job_batch
                            self.feat_job_batch[i_batch][
                                action_job, 0] = self.proc_time_batch[i_batch][action_job][action_ope]

                            # print("-1 ", self.schedule_batch[i_batch][mas_idx][-1][3])
                            # print(self.schedule_batch[i_batch])
                            self.schedule_batch[i_batch][mas_idx][-1][3] = self.time[i_batch].item()

                            add_time_batch[i_batch] = self.feat_arc_ma_out_batch[
                                i_batch, action_ope, action_station, 1]

                        else:
                            # update feat_buf_batch
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 0] -= 1
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 2] -= self.proc_time_batch[
                                i_batch][action_job][self.job_next_ope[i_batch][action_job]]
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 3] -= torch.sum(
                                self.proc_time_batch[i_batch][action_job, self.job_next_ope[i_batch][action_job]:])
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 4] = self.feat_buf_batch[
                                i_batch, action_station-self.station_num-3, 3] / torch.sum(torch.sum(
                                    self.proc_time_batch[i_batch][action_job, :]))
                            self.feat_buf_batch[i_batch, action_station-self.station_num-3, 5] = 0

                            if action_station == self.station_num + 1:
                                self.feat_ope_batch[i_batch, action_ope, 2] -= 1
                            if action_station > self.station_num:
                                self.schedule_batch[i_batch][
                                    action_station-self.station_num-3][-1][2] = self.time[i_batch].item()
                                self.schedule_batch[i_batch][
                                    action_station-self.station_num-3][-1][3] = self.time[i_batch].item()

                            add_time_batch[i_batch] = self.feat_arc_buf_out_batch[
                                i_batch, action_ope, action_station-self.station_num-3, 1]

                        # update job location information
                        self.job_loc_batch[i_batch][action_job, :] = 0
                        self.job_loc_ma_batch[i_batch][action_job, :] = 0
                        self.robot_loc_batch[i_batch, action_station] = 1
                        self.feat_ope_batch[i_batch, action_ope, 1] += 1

                        self.feat_arc_ma_in_batch[i_batch, :, :, 2] += 1
                        self.feat_arc_ma_out_batch[i_batch, :, :, 2] += 1
                        self.feat_arc_buf_in_batch[i_batch, :, :, 2] += 1
                        self.feat_arc_buf_out_batch[i_batch, :, :, 2] += 1

        self.done = self.done_batch.all()

        next_time_batch = add_time_batch + self.time

        # update the state information of the environment after time change
        for i_idxes in range(len(self.batch_idxes)):
            i_batch = self.batch_idxes[i_idxes]

            self.reward_batch[i_batch] -= torch.sum(self.mas_left_proctime_batch[i_batch, :])

            self.ope_node_job_batch[i_batch][:] = 0

            mas_state_change_idx = torch.nonzero((self.mas_left_proctime_batch[i_batch] <= add_time_batch[i_batch])
                                                 & (self.mas_left_proctime_batch[i_batch] > 0))

            # print(mas_state_change_idx)
            for i_change_mas_idx in mas_state_change_idx:
                # print(i_change_mas_idx)
                # print(self.time[i_batch])
                # print(self.mas_left_proctime_batch[i_batch, i_change_mas_idx])
                # print("fdas", self.schedule_batch[i_batch][i_change_mas_idx][-1])
                # print("fasdf", self.schedule_batch[i_batch])
                self.schedule_batch[i_batch][i_change_mas_idx][-1][2] = (self.time[
                        i_batch] + self.mas_left_proctime_batch[i_batch, i_change_mas_idx]).item()
                # print("fdsa", self.schedule_batch[i_batch])


            self.mas_state_batch[i_batch, mas_state_change_idx] = 2
            self.mas_left_proctime_batch[i_batch] -= add_time_batch[i_batch]
            self.mas_left_proctime_batch[i_batch] = torch.clamp(self.mas_left_proctime_batch[i_batch], min=0)

            self.reward_batch[i_batch] += torch.sum(self.mas_left_proctime_batch[i_batch, :])

            procing_job_idxes = torch.nonzero(self.job_state_batch[i_batch] == 1).squeeze()
            self.left_proc_time_batch[i_batch][
                procing_job_idxes, self.job_next_ope[i_batch][procing_job_idxes]] -= add_time_batch[i_batch]
            self.left_proc_time_batch[i_batch][procing_job_idxes] = torch.clamp(self.left_proc_time_batch[i_batch][
                procing_job_idxes], min=0)

            job_state_change_idx = torch.nonzero(
                self.job_loc_ma_batch[i_batch][:, mas_state_change_idx], as_tuple=True)[0]

            self.job_next_ope[i_batch][job_state_change_idx] += 1
            self.job_state_batch[i_batch][job_state_change_idx] = 2
            self.ope_job_batch[i_batch][:, job_state_change_idx] = 0
            self.ope_job_batch[i_batch][self.job_next_ope[i_batch][job_state_change_idx], job_state_change_idx] = 1

            job_on_robot_idx = torch.nonzero(torch.sum(self.job_loc_batch[i_batch], dim=1) == 0).squeeze(-1)
            self.ope_node_job_batch[i_batch][self.job_next_ope[i_batch][job_on_robot_idx], job_on_robot_idx, 0] = 1
            self.ope_node_job_batch[i_batch][self.job_next_ope[i_batch][job_on_robot_idx], job_on_robot_idx, 1] = 1
            for i_job_on_robot in job_on_robot_idx:
                if ~self.job_to_buf_flag_batch[i_batch][i_job_on_robot, self.job_next_ope[i_batch][i_job_on_robot]]:
                    if self.job_next_ope[i_batch][i_job_on_robot] < self.ope_num:
                        self.ope_node_job_batch[i_batch][
                            self.job_next_ope[i_batch][i_job_on_robot], i_job_on_robot, 1] = 0

                        num_idle_mas = 0
                        for idx_mas_last_ope in range(self.ope_num-1):
                            if self.routing[idx_mas_last_ope + 1] == self.routing[
                                                                            self.job_next_ope[i_batch][i_job_on_robot]]:
                                if 0 < self.feat_ope_batch[i_batch, idx_mas_last_ope, 6] <= self.proc_time_batch[
                                                i_batch][i_job_on_robot][self.job_next_ope[i_batch][i_job_on_robot]]:
                                    num_idle_mas += 1
                        if num_idle_mas <= self.feat_mas_batch[i_batch, self.routing[
                                                                    self.job_next_ope[i_batch][i_job_on_robot]]-1, 0]:
                            self.ope_node_job_batch[i_batch][
                                self.job_next_ope[i_batch][i_job_on_robot], i_job_on_robot, 1] = 0

            job_on_mas_finish_idx = self.job_loc_batch[i_batch] * (
                    self.job_state_batch[i_batch] == 2).long().unsqueeze(-1)
            job_on_mas_finish_idx[:, -1] = 0
            job_on_mas_finish_idx = torch.nonzero(torch.sum(job_on_mas_finish_idx, dim=1)).squeeze(-1)
            self.ope_node_job_batch[i_batch][
                self.job_next_ope[i_batch][job_on_mas_finish_idx], job_on_mas_finish_idx, 2] = 1
            job_on_buf_idle_idx = self.job_loc_batch[i_batch] * (
                    self.job_state_batch[i_batch] == 0).long().unsqueeze(-1)
            job_on_buf_idle_idx[:, -1] = 0
            job_on_buf_idle_idx = torch.nonzero(torch.sum(job_on_buf_idle_idx, dim=1)).squeeze(-1)
            self.ope_node_job_batch[i_batch][
                self.job_next_ope[i_batch][job_on_buf_idle_idx], job_on_buf_idle_idx, 3] = 1

            # update feat_mas_batch
            station_change_num = torch.zeros(size=(self.station_num,), device=self.device, dtype=torch.long)
            station_mas = torch.repeat_interleave(torch.arange(self.station_num), self.paral_mas_num)
            for i_mas in mas_state_change_idx:
                station_change_num[station_mas[i_mas]] += 1

            self.feat_mas_batch[i_batch, :, 1] += station_change_num

            ope_job_num = torch.sum(self.ope_job_batch[i_batch], dim=1)
            # size=(num_job, num_ope+1)

            current_ope_time = self.left_proc_time_batch[i_batch] * self.ope_job_batch[i_batch].T

            for i_station in range(self.station_num):
                # update self.feat_mas_batch[i_batch, :, 2]

                i_station_left_proc_time = self.mas_left_proctime_batch[
                          i_batch, (self.mas_appertain_batch[i_station]):(
                                self.mas_appertain_batch[i_station] + self.paral_mas_num[i_station])]
                i_station_left_proc_time_positive = i_station_left_proc_time[i_station_left_proc_time > 0]
                self.feat_mas_batch[i_batch, i_station, 2] = torch.min(
                    i_station_left_proc_time_positive) if i_station_left_proc_time_positive.nelement() > 0 else 0

                self.feat_mas_batch[i_batch, i_station, 3] = torch.sum(
                    ope_job_num[torch.nonzero(self.routing == i_station+1)])

                self.feat_mas_batch[i_batch, i_station, 4] = torch.sum(
                    torch.sum(current_ope_time, dim=0)[torch.nonzero(self.routing == i_station+1)])

            # update feat_opes_batch
            ope_change_idx = torch.nonzero(self.ope_job_batch[i_batch][:, job_state_change_idx], as_tuple=True)[0]
            for ope_idx in ope_change_idx:
                self.feat_ope_batch[i_batch, ope_idx, 0] += 1

            self.feat_ope_batch[i_batch, :, 3] = 1 - torch.sum(self.left_proc_time_batch[i_batch], dim=0) / torch.sum(
                self.proc_time_batch[i_batch], dim=0) \
                if torch.sum(self.proc_time_batch[i_batch], dim=0).nelement() > 0 else 0
            self.feat_ope_batch[i_batch, :, 4] = torch.sum(self.left_proc_time_batch[i_batch], dim=0)
            self.feat_ope_batch[i_batch, :, 5] = torch.sum(self.left_proc_time_batch[i_batch] * (
                    self.job_state_batch[i_batch] == 0).float().unsqueeze(-1) * self.ope_job_batch[i_batch].T, dim=0)

            min_ope_procing_time = self.left_proc_time_batch[i_batch] * (
                    self.job_state_batch[i_batch] == 1).float().unsqueeze(-1) * self.ope_job_batch[i_batch].T
            min_ope_procing_time = min_ope_procing_time.masked_fill(min_ope_procing_time == 0., float('inf'))
            self.feat_ope_batch[i_batch, :, 6] = torch.min(min_ope_procing_time, dim=0).values
            self.feat_ope_batch[i_batch, :, 6][self.feat_ope_batch[i_batch, :, 6] == float('inf')] = 0.

            # update feat_job_batch
            self.feat_job_batch[i_batch][:, 0] = torch.sum(
                self.left_proc_time_batch[i_batch] * self.ope_job_batch[i_batch].T, dim=1)
            self.feat_job_batch[i_batch][:, 1] = torch.sum(self.left_proc_time_batch[i_batch], dim=1)
            self.feat_job_batch[i_batch][:, 1] = torch.sum((self.left_proc_time_batch[i_batch] > 0).long(), dim=1)

            # update feat_arc_batch
            for i_ope in range(self.ope_num):
                self.feat_arc_ma_in_batch[i_batch, i_ope, :, 0] = torch.mean(self.proc_time_batch[i_batch][
                    torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 0]).squeeze(-1)][:, i_ope]) \
                    if torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 0]).size(0) != 0 else 0
                self.feat_arc_ma_out_batch[i_batch, i_ope, :, 0] = torch.mean(self.proc_time_batch[i_batch][
                    torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 2]).squeeze(-1)][:, i_ope]) \
                    if torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 2]).size(0) != 0 else 0
                self.feat_arc_buf_in_batch[i_batch, i_ope, :, 0] = torch.mean(self.proc_time_batch[i_batch][
                    torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 1]).squeeze(-1)][:, i_ope]) \
                    if torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 1]).size(0) != 0 else 0
                self.feat_arc_buf_out_batch[i_batch, i_ope, :, 0] = torch.mean(self.proc_time_batch[i_batch][
                    torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 3]).squeeze(-1)][:, i_ope]) \
                    if torch.nonzero(self.ope_node_job_batch[i_batch][i_ope, :, 3]).size(0) != 0 else 0

            self.feat_arc_ma_in_batch[i_batch, :, :, 1] = self.trans_time[torch.nonzero(
                self.robot_loc_batch[i_batch, :]).squeeze(-1), None, :].squeeze(0).T + self.unload_time
            self.feat_arc_ma_in_batch[i_batch, :, :, 2] = torch.sum(self.feat_ope_batch[i_batch, :, 1])

            self.feat_arc_ma_out_batch[i_batch, :, :, 1] = self.trans_time[torch.nonzero(
                self.robot_loc_batch[i_batch, :]).squeeze(-1), None, :].squeeze(0).T + self.load_time
            self.feat_arc_ma_out_batch[i_batch, :, :, 2] = torch.sum(self.feat_ope_batch[i_batch, :, 1])

            self.mask_mas_arc_batch[i_batch] = False
            self.mask_buf_arc_batch[i_batch] = False
            self.mask_wait_batch[i_batch] = True
            self.mask_job_batch[i_batch][:] = False
            for i_job_on_robot in job_on_robot_idx:
                if self.job_next_ope[i_batch][i_job_on_robot] < self.ope_num:
                    if self.feat_mas_batch[i_batch, self.routing[self.job_next_ope[i_batch][i_job_on_robot]] - 1, 0] > 0:
                        self.mask_mas_arc_batch[i_batch, self.job_next_ope[i_batch][i_job_on_robot], self.routing[
                            self.job_next_ope[i_batch][i_job_on_robot]] - 1, 0] = True
                        self.mask_job_batch[i_batch][i_job_on_robot] = True
                # if i_job_on_robot == 3:
                    # print(i_job_on_robot)
                    # print(job_on_robot_idx)
                    # print("a", self.feat_buf_batch[i_batch, 1, 0] < self.buffer_cap)
                    # print("a", self.job_to_buf_flag_batch[i_batch][i_job_on_robot, self.job_next_ope[i_batch][i_job_on_robot]])
                if self.feat_buf_batch[i_batch, 1, 0] < self.buffer_cap and self.job_to_buf_flag_batch[i_batch][
                    i_job_on_robot, self.job_next_ope[i_batch][i_job_on_robot]] and self.ope_node_job_batch[i_batch][
                                                    self.job_next_ope[i_batch][i_job_on_robot]][i_job_on_robot][1] == 1:
                    self.mask_buf_arc_batch[i_batch, self.job_next_ope[i_batch][i_job_on_robot], 1, 0] = True
                    self.mask_job_batch[i_batch][i_job_on_robot] = True
                    # num_idle_mas = self.feat_mas_batch[
                        # i_batch, self.routing[self.job_next_ope[i_batch][i_job_on_robot]]-1, 0].item()
                    # num_idle_mas = 0
                    # for idx_mas_last_ope in range(self.ope_num-1):
                    #     if self.routing[idx_mas_last_ope + 1] == self.routing[self.job_next_ope[i_batch][i_job_on_robot]]:
                    #         if 0 < self.feat_ope_batch[i_batch, idx_mas_last_ope, 6] <= self.proc_time_batch[i_batch][i_job_on_robot][self.job_next_ope[i_batch][i_job_on_robot]]:
                    #             num_idle_mas += 1
                    # if num_idle_mas < self.feat_mas_batch[i_batch, self.routing[self.job_next_ope[i_batch][i_job_on_robot]]-1, 0]:
                    #     self.mask_buf_arc_batch[i_batch, self.job_next_ope[i_batch][i_job_on_robot], 1, 0] = True

                if self.job_next_ope[i_batch][i_job_on_robot] == self.ope_num:
                    self.mask_buf_arc_batch[i_batch, self.job_next_ope[i_batch][i_job_on_robot], 2, 0] = True
                    self.mask_job_batch[i_batch][i_job_on_robot] = True



            # in-buffer limit
            # for i_ope_buffer in range(self.ope_num-1):
                # if self.mask_buf_arc_batch[i_batch, i_ope_buffer, 1, 0]:


            # policy for avoiding deadlock
            if job_on_robot_idx.size(0) < self.trans_cap:
                for i_job_on_mas_finish in job_on_mas_finish_idx:
                    self.mask_mas_arc_batch[i_batch, self.job_next_ope[i_batch][i_job_on_mas_finish], self.routing[
                        self.job_next_ope[i_batch][i_job_on_mas_finish]-1] - 1, 1] = True
                    self.mask_job_batch[i_batch][i_job_on_mas_finish] = True
                for i_job_on_buf_idle in job_on_buf_idle_idx:
                    self.mask_buf_arc_batch[i_batch, self.job_next_ope[i_batch][i_job_on_buf_idle], torch.nonzero(
                        self.job_loc_batch[i_batch][i_job_on_buf_idle, :]).squeeze(-1)-self.station_num, 1] = True
                    self.mask_job_batch[i_batch][i_job_on_buf_idle] = True
                    # Add action mask to overcome deadlocks
                    if job_on_robot_idx.size(0) > self.trans_cap - 2:
                        flag_mask = True
                        for i_job_on_robot in job_on_robot_idx:
                            if self.job_next_ope[i_batch][i_job_on_robot] < self.ope_num:
                                if self.feat_mas_batch[i_batch, self.routing[
                                                                self.job_next_ope[i_batch][i_job_on_robot]] - 1, 0] > 0:
                                    flag_mask = False
                                # elif self.job_to_buf_flag_batch[
                                        # i_batch][i_job_on_robot, self.job_next_ope[i_batch][i_job_on_robot]]:
                                elif self.ope_node_job_batch[i_batch][self.job_next_ope[i_batch][i_job_on_robot]][
                                                    i_job_on_robot][1] == 1 and self.job_to_buf_flag_batch[i_batch][
                                                            i_job_on_robot, self.job_next_ope[i_batch][i_job_on_robot]]:
                                    flag_mask = False
                            if ~flag_mask:
                                break

                        if flag_mask:
                            if self.feat_mas_batch[i_batch, self.routing[
                                                            self.job_next_ope[i_batch][i_job_on_buf_idle]] - 1, 0] < 1:
                                self.mask_buf_arc_batch[
                                    i_batch, self.job_next_ope[i_batch][i_job_on_buf_idle], torch.nonzero(
                                        self.job_loc_batch[i_batch][i_job_on_buf_idle, :]).squeeze(
                                        -1) - self.station_num, 1] = False

            if torch.sum(self.feat_mas_batch[i_batch, :, 2]) == 0.:
                self.mask_wait_batch[i_batch] = False

            self.time[i_batch] = next_time_batch[i_batch]
            self.reward_batch[i_batch] = self.reward_batch[i_batch] / torch.sum(self.proc_time_batch[i_batch])
            self.cum_reward_batch[i_batch] += self.reward_batch[i_batch]

        batch_idxes_update = self.batch_idxes[~self.done_batch[self.batch_idxes]]
        self.batch_idxes = batch_idxes_update

        self.state.update(batch_idxes=self.batch_idxes, mas_state_batch=self.mas_state_batch,
                          mas_left_proctime_batch=self.mas_left_proctime_batch, feat_ope_batch=self.feat_ope_batch,
                          feat_mas_batch=self.feat_mas_batch, feat_buf_batch=self.feat_buf_batch,
                          feat_arc_ma_in_batch=self.feat_arc_ma_in_batch,
                          feat_arc_ma_out_batch=self.feat_arc_ma_out_batch,
                          feat_arc_buf_in_batch=self.feat_arc_buf_in_batch,
                          feat_arc_buf_out_batch=self.feat_arc_buf_out_batch,
                          feat_job_batch=self.feat_job_batch, job_next_ope=self.job_next_ope,
                          ope_job_batch=self.ope_job_batch, job_loc_batch=self.job_loc_batch,
                          job_loc_ma_batch=self.job_loc_ma_batch, done_job_batch=self.done_job_batch,
                          ope_node_job_batch=self.ope_node_job_batch, num_jobs_batch=self.num_jobs_batch,
                          robot_loc_batch=self.robot_loc_batch, mask_mas_arc_batch=self.mask_mas_arc_batch,
                          mask_buf_arc_batch=self.mask_buf_arc_batch, job_state_batch=self.job_state_batch,
                          mask_job_batch=self.mask_job_batch, mask_wait_batch=self.mask_wait_batch)

        return self.state, self.reward_batch, self.cum_reward_batch, self.done_batch

    def reset(self):
        """
        Reset the environment
        """
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = self.state.batch_idxes
        self.mas_state_batch = self.state.mas_state_batch
        self.mas_left_proctime_batch = self.state.mas_left_proctime_batch
        self.feat_ope_batch = self.state.feat_ope_batch
        self.feat_mas_batch = self.state.feat_mas_batch
        self.feat_buf_batch = self.state.feat_buf_batch
        self.feat_arc_ma_in_batch = self.state.feat_arc_ma_in_batch
        self.feat_arc_ma_out_batch = self.state.feat_arc_ma_out_batch
        self.feat_arc_buf_in_batch = self.state.feat_arc_buf_in_batch
        self.feat_arc_buf_out_batch = self.state.feat_arc_buf_out_batch
        self.feat_job_batch = self.state.feat_job_batch
        self.job_next_ope = self.state.job_next_ope
        self.ope_job_batch = self.state.ope_job_batch
        self.job_loc_batch = self.state.job_loc_batch
        self.job_loc_ma_batch = self.state.job_loc_ma_batch
        self.ope_node_job_batch = self.state.ope_node_job_batch

        self.num_jobs_batch = self.state.num_jobs_batch
        self.robot_loc_batch = self.state.robot_loc_batch
        self.job_state_batch = self.state.job_state_batch
        self.done_job_batch = self.state.done_job_batch
        self.mask_mas_arc_batch = self.state.mask_mas_arc_batch
        self.mask_buf_arc_batch = self.state.mask_buf_arc_batch
        self.mask_job_batch = self.state.mask_job_batch
        self.mask_wait_batch = self.state.mask_wait_batch

        schedule_batch = []
        for i_batch_case in range(self.batch_size):
            schedule = []
            for _ in range(self.mas_num + 3):
                schedule.append([])
            schedule_batch.append(schedule)
        self.schedule_batch = schedule_batch

        self.time = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)
        self.makespan_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)
        self.done_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.bool)
        self.done = self.done_batch.all()
        self.reward_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)
        self.cum_reward_batch = torch.zeros(size=(self.batch_size,), device=self.device, dtype=torch.float)

    def render(self, mode='human'):
        """
        Render the environment
        """
        color_path = os.path.join(os.getcwd(), "color_config.json")
        with open(color_path, "r", encoding="utf-8") as f:
            color = json.load(f)["gantt_color"]

        for batch_id in range(self.batch_size):
            num_jobs = self.num_jobs_batch[batch_id]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            with open(color_path, 'w', encoding='UTF-8') as fp:
                fp.write(json.dumps({"gantt_color": color}, indent=2, ensure_ascii=False))

            schedule = self.schedule_batch[batch_id]
            fig = plt.figure()
            fig.canvas.manager.set_window_title('Gantt Chart')
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            y_ticks = []
            y_ticks_loc = []
            for i in range(self.mas_num):
                y_ticks.append('M {0}'.format(i))
                y_ticks_loc.append(i)
            labels = [''] * num_jobs
            for j in range(num_jobs):
                labels[j] = 'J {0}'.format(j)
            patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(num_jobs)]
            axes.cla()
            axes.grid(linestyle='-.', color='gray', alpha=0.3)
            axes.set_xlabel("Time")
            axes.set_ylabel("Machine")
            axes.set_yticks(y_ticks_loc, y_ticks)
            axes.legend(handles=patches, loc=2)
            axes.set_ybound(1 - 1 / self.mas_num, self.mas_num + 1 / self.mas_num)
            # axes.set_xlim(0, 550)

            for i_mas in range(self.mas_num):
                # print(schedule[i_mas])
                for proc_idx in schedule[i_mas]:
                    # print(proc_idx)
                    axes.barh(i_mas,
                              proc_idx[2] - proc_idx[1],
                              left=proc_idx[1],
                              color=color[proc_idx[0]],
                              edgecolor=color[proc_idx[0]],
                              alpha=1,
                              height=0.5)
                    axes.barh(i_mas,
                              proc_idx[3] - proc_idx[2],
                              left=proc_idx[2],
                              color=color[proc_idx[0]],
                              edgecolor=color[proc_idx[0]],
                              alpha=0.3,
                              height=0.5)

        plt.show()

        return

    def validate_gantt(self):
        """
        Validate whether the schedule is feasible
        """
        pass

    def close(self):
        """
        Close the environment
        """
        pass


if __name__ == "__main__":
    with open("./shop_info_paras.json", 'r') as load_f:
        load_dict_shop = json.load(load_f)
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    load_dict_env = load_dict["env_paras"]

    path = os.path.abspath('.') + "/data/"

    shop_info = shop_info_initial(load_dict_shop)
    # case = CaseGenerator(shop_info, path=path, flag_doc=True)
    # for i_instance in range(3):
    #     instance, num_jobs = case.get_case(i_instance)
    # a = RRPFSPEnv(case=None, shop_info=shop_info, env_paras=load_dict_env, data_source='case')
    valid_data_flies = os.listdir(path)
    valid_data_flies.sort()
    for i_valid in range(len(valid_data_flies)):
        valid_data_flies[i_valid] = path + valid_data_flies[i_valid]

    env_valid = RRPFSPEnv(case=valid_data_flies, shop_info=shop_info, env_paras=load_dict_env, data_source='file')
    arc_actions = torch.zeros(size=(load_dict_env["batch_size"], shop_info.ope_num+1, shop_info.station_num+3, 2))
    arc_actions[:, 0, env_valid.station_num, 1] = 1
    # arc_actions[0, 0, env_valid.station_num, 1] = 0
    job_actions = torch.zeros(size=(load_dict_env["batch_size"],), dtype=torch.long)
    job_actions[1] = 1
    job_actions[2] = 4
    actions_0 = [arc_actions, job_actions]
    env_valid.step(actions_0)

    arc_actions = torch.zeros(size=(load_dict_env["batch_size"], shop_info.ope_num + 1, shop_info.station_num + 3, 2))
    arc_actions[:, 0, env_valid.routing[0]-1, 0] = 1
    job_actions = torch.zeros(size=(load_dict_env["batch_size"],), dtype=torch.long)
    actions_1 = [arc_actions, job_actions]
    env_valid.step(actions_1)








