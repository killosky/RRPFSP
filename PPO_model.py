

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from itertools import chain

from graph.hgnn import GATedge, MLPsim
from mlp import MLPActor, MLPCritic, MLPJob
from utils import transpose_list_of_tensors, memory_flatten


class Memory:
    def __init__(self):
        self.rewards = []
        self.logprobs = []
        self.logprobs_job = []
        self.rewards = []
        self.is_terminals = []
        self.action_envs = []
        self.action_job_envs = []
        self.batch_idxes = []

        self.ope_ma_adj = []
        self.ope_ma_adj_out = []
        self.ope_buf_adj = []
        self.ope_buf_adj_out = []
        self.raw_opes = []
        self.raw_mas = []
        self.raw_buf = []
        self.raw_arc_ma_in = []
        self.raw_arc_ma_out = []
        self.raw_arc_buf_in = []
        self.raw_arc_buf_out = []
        self.raw_job = []
        self.eligible = []
        self.eligible_wait = []
        self.action_envs = []
        self.action_job_envs = []

    def clear_memory(self):
        del self.rewards[:]
        del self.logprobs[:]
        del self.logprobs_job[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_envs[:]
        del self.action_job_envs[:]
        del self.batch_idxes[:]

        del self.ope_ma_adj[:]
        del self.ope_ma_adj_out[:]
        del self.ope_buf_adj[:]
        del self.ope_buf_adj_out[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.raw_buf[:]
        del self.raw_arc_ma_in[:]
        del self.raw_arc_ma_out[:]
        del self.raw_arc_buf_in[:]
        del self.raw_arc_buf_out[:]
        del self.raw_job[:]
        del self.eligible[:]
        del self.eligible_wait[:]
        del self.action_envs[:]
        del self.action_job_envs[:]


class MLPs(nn.Module):
    """
    MLPs for operation node embedding
    """
    def __init__(self, in_sizes_ope_embed, hidden_size_ope, out_size_ope, num_head, dropout, device=torch.device('cuda')):
        """
        :param in_sizes_ope_embed: A list of the dimensions of input vector for each type,
        including [operation(pre), operation(sub), machine(in), machine(out), buffer, operation(self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        """
        super(MLPs, self).__init__()
        self.in_sizes_ope_embed = in_sizes_ope_embed
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.device = device

        self.gnn_layers = nn.ModuleList()

        for i_feat in range(len(self.in_sizes_ope_embed)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope_embed[i_feat],
                                          self.out_size_ope, self.hidden_size_ope, self.num_head).to(self.device))
        self.project = nn.Sequential(
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope_embed), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope)
        )

    def forward(self, adj, batch_idxes, feats):
        """
        :param adj: (ope_ma_adj, ope_ma_adj_out, ope_buf_adj, ope_buf_adj_out)
        :param batch_idxes: batch index
        :param feats: (feat_opes_batch, feat_mas_batch, feat_buf_batch)
        """

        # ope adj matrix
        ope_self_adj = torch.eye(
            feats[0].size(-2), dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_idxes.size(0), -1, -1)
        ope_pre_adj = torch.diag(torch.ones(feats[0].size(-2)-1, dtype=torch.long, device=self.device), 1) \
            .unsqueeze(0).expand(batch_idxes.size(0), -1, -1)
        ope_sub_adj = torch.transpose(ope_pre_adj, -2, -1)

        h = (feats[0], feats[0], feats[1], feats[1], feats[2], feats[0])

        ope_buf_adj_or = torch.logical_or(
            adj[2].bool(), adj[3].bool()).long().unsqueeze(0).expand(batch_idxes.size(0), -1, -1)
        adj_matrix = (ope_pre_adj, ope_sub_adj, adj[0].unsqueeze(0).expand(batch_idxes.size(0), -1, -1),
                      adj[1].unsqueeze(0).expand(batch_idxes.size(0), -1, -1), ope_buf_adj_or, ope_self_adj)

        MLP_embeddings = []
        for i_feat in range(len(self.in_sizes_ope_embed)):
            MLP_embeddings.append(self.gnn_layers[i_feat](h[i_feat], adj_matrix[i_feat]))

        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_i_prime = self.project(MLP_embedding_in)

        return mu_i_prime


def get_normalized(raw_opes, raw_mas, raw_buf, raw_arc_ma_in, raw_arc_ma_out,
                   raw_arc_buf_in, raw_arc_buf_out, raw_job, batch_idxes):
    """
    Normalized time variable in feats, including operations, machines and arcs feats.
    """
    mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
    std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
    mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
    std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
    mean_buf = torch.mean(raw_buf, dim=-2, keepdim=True)
    std_buf = torch.std(raw_buf, dim=-2, keepdim=True)
    mean_arc_ma_in = torch.mean(raw_arc_ma_in, dim=(1, 2), keepdim=True)
    std_arc_ma_in = torch.std(raw_arc_ma_in, dim=(1, 2), keepdim=True)
    mean_arc_ma_out = torch.mean(raw_arc_ma_out, dim=(1, 2), keepdim=True)
    std_arc_ma_out = torch.std(raw_arc_ma_out, dim=(1, 2), keepdim=True)
    mean_arc_buf_in = torch.mean(raw_arc_buf_in, dim=(1, 2), keepdim=True)
    std_arc_buf_in = torch.std(raw_arc_buf_in, dim=(1, 2), keepdim=True)
    mean_arc_buf_out = torch.mean(raw_arc_buf_out, dim=(1, 2), keepdim=True)
    std_arc_buf_out = torch.std(raw_arc_buf_out, dim=(1, 2), keepdim=True)

    feat_job_normalized = []
    for i_idxes in range(len(batch_idxes)):
        mean_jobs = torch.mean(raw_job[batch_idxes[i_idxes]], dim=-2, keepdim=True)
        std_jobs = torch.std(raw_job[batch_idxes[i_idxes]], dim=-2, keepdim=True)
        feat_job_normalized.append((raw_job[batch_idxes[i_idxes]] - mean_jobs) / (std_jobs + 1e-5))

    return ((raw_opes - mean_opes) / (std_opes + 1e-5),
            (raw_mas - mean_mas) / (std_mas + 1e-5),
            (raw_buf - mean_buf) / (std_buf + 1e-5),
            (raw_arc_ma_in - mean_arc_ma_in) / (std_arc_ma_in + 1e-5),
            (raw_arc_ma_out - mean_arc_ma_out) / (std_arc_ma_out + 1e-5),
            (raw_arc_buf_in - mean_arc_buf_in) / (std_arc_buf_in + 1e-5),
            (raw_arc_buf_out - mean_arc_buf_out) / (std_arc_buf_out + 1e-5),
            feat_job_normalized)


class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()

        self.device = torch.device(model_paras['device'])

        self.in_size_ma = model_paras['in_size_ma']    # Dimension of machine and buffer features
        self.in_size_ope = model_paras['in_size_ope']    # Dimension of operation features
        self.in_size_arc = model_paras['in_size_arc']    # Dimension of arc features
        self.in_size_job = model_paras['in_size_job']    # Dimension of job features

        self.out_size_ma = model_paras['out_size_ma']    # Dimension of machine/buffer embedding in GAT edge
        self.out_size_ope = model_paras['out_size_ope']    # Dimension of operation embedding in MLPs
        self.hidden_size_ope = model_paras['hidden_size_ope']    # Hidden dimensions of the MLPs

        self.actor_dim = model_paras['actor_in_dim']    # Input dimension of the actor network
        self.critic_dim = model_paras['critic_in_dim']    # Input dimension of the critic network
        self.job_section_dim = model_paras['job_selection_dim']    # Input dimension of the job section network

        self.n_hidden_actor = model_paras['n_hidden_actor']  # Hidden dimension of the actor network
        self.n_hidden_critic = model_paras['n_hidden_critic']  # Hidden dimension of the critic network
        self.n_layers_actor = model_paras['n_layers_actor']  # Number of layers of the actor network
        self.n_layers_critic = model_paras['n_layers_critic']  # Number of layers of the critic network

        self.n_layers_hgnn = model_paras['n_layers_hgnn']  # Number of layers of the HGNN GATedge

        self.n_layers_job = model_paras['n_layers_job']    # Number of layers of the job section network
        self.n_hidden_job = model_paras['n_hidden_job']    # Hidden dimension of the job section network

        self.action_dim = model_paras['action_dim']  # Output dimension of the action space
        self.num_head = model_paras['num_head']  # Number of heads in GATedge
        self.dropout = model_paras['dropout']  # Dropout rate in GATedge

        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge(in_feats=(self.in_size_ope, self.in_size_ma, self.in_size_arc),
                                         out_feats=self.out_size_ma, num_head=self.num_head[0], feat_drop=self.dropout,
                                         attn_drop=self.dropout, device=self.device).to(self.device))
        for i_layer_hgnn in range(self.n_layers_hgnn - 1):
            self.get_machines.append(GATedge(in_feats=(self.out_size_ope, self.out_size_ma, self.in_size_arc),
                                             out_feats=self.out_size_ma, num_head=self.num_head[0],
                                             feat_drop=self.dropout, attn_drop=self.dropout, device=self.device).to(self.device))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs(in_sizes_ope_embed=[
            self.in_size_ope, self.in_size_ope, self.out_size_ma, self.out_size_ma, self.out_size_ma, self.in_size_ope],
            hidden_size_ope=self.hidden_size_ope, out_size_ope=self.out_size_ope, num_head=self.num_head[0],
            dropout=self.dropout, device=self.device).to(self.device))
        for i_layer_hgnn in range(self.n_layers_hgnn - 1):
            self.get_operations.append(MLPs(
                in_sizes_ope_embed=[self.out_size_ope, self.out_size_ope, self.out_size_ma, self.out_size_ma,
                                    self.out_size_ma, self.out_size_ope],
                hidden_size_ope=self.hidden_size_ope, out_size_ope=self.out_size_ope, num_head=self.num_head[0],
                dropout=self.dropout, device=self.device).to(self.device))

        # Actor and critic networks
        self.actor = MLPActor(self.n_layers_actor, self.actor_dim, self.n_hidden_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_layers_critic, self.critic_dim, self.n_hidden_critic, 1).to(self.device)

        # Job section network
        self.get_jobs = MLPJob(self.n_layers_job, self.job_section_dim, self.n_hidden_job, 1).to(self.device)

    def forward(self):
        raise NotImplementedError

    def get_arc_prob(self, state, memories, flag_train=True):
        """
        Get the probability of each arc in decision-making
        """
        # Uncompleted instances
        batch_idxes = state.batch_idxes

        # Raw features
        raw_opes = state.feat_ope_batch[batch_idxes]
        raw_mas = state.feat_mas_batch[batch_idxes]
        raw_buf = state.feat_buf_batch[batch_idxes]
        raw_arc_ma_in = state.feat_arc_ma_in_batch[batch_idxes]
        raw_arc_ma_out = state.feat_arc_ma_out_batch[batch_idxes]
        raw_arc_buf_in = state.feat_arc_buf_in_batch[batch_idxes]
        raw_arc_buf_out = state.feat_arc_buf_out_batch[batch_idxes]

        # Normalized input features
        opes_norm, mas_norm, buf_norm, arc_ma_in_norm, arc_ma_out_norm, arc_buf_in_norm, arc_buf_out_norm, job_norm = \
            get_normalized(raw_opes, raw_mas, raw_buf, raw_arc_ma_in, raw_arc_ma_out, raw_arc_buf_in, raw_arc_buf_out,
                           state.feat_job_batch, batch_idxes)

        # L iterations of HGNN
        features = (opes_norm, mas_norm, buf_norm, arc_ma_in_norm, arc_buf_in_norm, arc_ma_out_norm, arc_buf_out_norm)
        adj = (state.ope_ma_adj, state.ope_ma_adj_out, state.ope_buf_adj, state.ope_buf_adj_out)
        for i_layer_hgnn in range(self.n_layers_hgnn):
            # Machine node embedding
            h_mas, h_buf = self.get_machines[i_layer_hgnn](adj, batch_idxes, features)
            features = (features[0], h_mas, h_buf, features[3], features[4], features[5], features[6])
            # Operation node embedding
            h_opes = self.get_operations[i_layer_hgnn](adj, batch_idxes, features)
            features = (h_opes, features[1], features[2], features[3], features[4], features[5], features[6])

        # Stacking and polling
        # Average pooling of the machine embedding node with shape (batch_size, out_size_ma)
        h_mas_pooled = torch.mean(h_mas, dim=-2)
        # Average pooling of the buffer embedding node with shape (batch_size, out_size_ma)
        h_buf_pooled = torch.mean(h_buf, dim=-2)
        # Average pooling of the machine and buffer embedding node with shape (batch_size, out_size_ma)
        h_mas_buf_pooled = torch.mean(torch.cat((h_mas, h_buf), dim=-2), dim=-2)
        # Average pooling of the operation embedding node with shape (batch_size, out_size_ope)
        h_opes_pooled = torch.mean(h_opes, dim=-2)

        # Average polling of the job embedding node with shape (batch_size, in_size_job)
        h_job_pooled = torch.zeros(size=(len(batch_idxes), self.in_size_job), dtype=torch.float, device=self.device)
        for i_idxes in range(len(batch_idxes)):
            h_job_pooled[i_idxes] = torch.mean(job_norm[i_idxes], dim=-2)

        # arc_action: (len(batch idx), num_opes + 1, num_stations + 3, 2)

        # Get the eligible mask of the arc selection with shape (len(batch idx), num_opes + 1, num_stations + 3, 2)
        eligible = torch.cat((state.mask_mas_arc_batch[batch_idxes], state.mask_buf_arc_batch[batch_idxes]), dim=2)
        eligible_wait = state.mask_wait_batch[batch_idxes].unsqueeze(-1)    # size: (len(batch idx), 1)

        # Structure the tensor with the same dimension
        h_opes_padding = h_opes.unsqueeze(-2).expand(-1, -1, h_mas.size(-2)+h_buf.size(-2), -1)
        h_mas_buf_padding = torch.cat((h_mas, h_buf), dim=-2).unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_opes_padding)
        # h_mas_buf_pooled_padding = torch.cat((h_mas_pooled[:, None, None, :].expand_as(h_mas_buf_padding),
        #                                       h_buf_pooled[:, None, None, :].expand_as(h_mas_buf_padding)), dim=-1)
        h_mas_buf_pooled_padding = h_mas_buf_pooled[:, None, None, :].expand_as(h_mas_buf_padding)

        # Input of the actor network
        h_actions = torch.cat((
            h_opes_padding, h_mas_buf_padding, h_opes_pooled_padding, h_mas_buf_pooled_padding), dim=-1)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled, h_buf_pooled, h_job_pooled), dim=-1)

        # Get probability of actions with masking the ineligible actions
        scores = self.actor.forward(h_actions)
        scores[:, :, :, :2] = scores[:, :, :, :2].masked_fill(eligible == False, float('-inf'))
        # no_wait_action_scores = torch.max(scores[:, :, :, 0], scores[:, :, :, 1])
        # no_wait_action_scores = no_wait_action_scores.view(len(batch_idxes), -1)
        wait_action_scores = torch.mean(scores[:, :, :, 2], dim=(-1, -2)).unsqueeze(-1)
        wait_action_scores = wait_action_scores.masked_fill(eligible_wait == False, float('-inf'))
        # size: (len(batch_idxes), ope_num+1 * station_num+3 * 2 + 1)
        action_scores = torch.cat(
            (scores[:, :, :, 0].flatten(1), scores[:, :, :, 1].flatten(1), wait_action_scores), dim=-1)
        # action_scores = torch.cat((no_wait_action_scores, wait_action_scores), dim=-1)
        action_probs = F.softmax(action_scores, dim=-1)

        if flag_train:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj))
            memories.ope_ma_adj_out.append(copy.deepcopy(state.ope_ma_adj_out))
            memories.ope_buf_adj.append(copy.deepcopy(state.ope_buf_adj))
            memories.ope_buf_adj_out.append(copy.deepcopy(state.ope_buf_adj_out))
            memories.batch_idxes.append(copy.deepcopy(batch_idxes))
            memories.raw_opes.append(copy.deepcopy(opes_norm))
            memories.raw_mas.append(copy.deepcopy(mas_norm))
            memories.raw_buf.append(copy.deepcopy(buf_norm))
            memories.raw_arc_ma_in.append(copy.deepcopy(arc_ma_in_norm))
            memories.raw_arc_ma_out.append(copy.deepcopy(arc_ma_out_norm))
            memories.raw_arc_buf_in.append(copy.deepcopy(arc_buf_in_norm))
            memories.raw_arc_buf_out.append(copy.deepcopy(arc_buf_out_norm))
            memories.raw_job.append(copy.deepcopy(job_norm))
            memories.eligible.append(copy.deepcopy(eligible))
            memories.eligible_wait.append(copy.deepcopy(eligible_wait))

        return action_probs, h_pooled, job_norm

    def get_job_prob(self, state, job_norm, arc_action):
        """
        Get the job idx corresponding to each arc selection in each batch
        :param state: state of the environment
        :param job_norm: normalized job features
        :param arc_action: size=(num_opes + 1, num_stations + 3, 2)
        :return: job idx of the arc
        """
        # job_action = torch.zeros(size=(len(state.batch_idxes),), dtype=torch.long, device=self.device)
        action_job_probs = []

        for i_idxes in range(len(state.batch_idxes)):
            arc_idx = torch.nonzero(arc_action[i_idxes]).squeeze()
            if arc_idx.size(0) > 0:
                if arc_idx[-1] == 0:
                    if arc_idx[1] < state.station_num:
                        selection_job_mask = state.ope_node_job_batch[
                                                 state.batch_idxes[i_idxes]][arc_idx[0], :, 0].squeeze()
                    else:
                        selection_job_mask = state.ope_node_job_batch[
                                                 state.batch_idxes[i_idxes]][arc_idx[0], :, 1].squeeze()
                else:
                    if arc_idx[1] < state.station_num:
                        selection_job_mask = state.ope_node_job_batch[
                                                 state.batch_idxes[i_idxes]][arc_idx[0], :, 2].squeeze()
                    else:
                        selection_job_mask = state.ope_node_job_batch[
                                                 state.batch_idxes[i_idxes]][arc_idx[0], :, 3].squeeze()

                selection_job_mask = selection_job_mask * state.mask_job_batch[state.batch_idxes[i_idxes]].long()

                score_job = self.get_jobs.forward(job_norm[i_idxes])
                score_job = score_job.squeeze(-1)

                score_job = score_job.masked_fill(selection_job_mask == 0, float('-inf'))
                action_job_prob = F.softmax(score_job, dim=-1)

            else:
                action_job_prob = torch.ones(size=(
                    state.num_jobs_batch[state.batch_idxes[i_idxes]],), dtype=torch.float, device=self.device)
                action_job_prob = F.softmax(action_job_prob, dim=-1)

            action_job_probs.append(action_job_prob)

        return action_job_probs

    def act(self, state, memories, done, flag_sample=True, flag_train=True):
        """
        Get actions of the env
        """
        action_probs, _, job_norm = self.get_arc_prob(state, memories, flag_train)
        dist = Categorical(action_probs)

        if flag_sample:
            action_index = dist.sample()
        else:
            action_index = torch.argmax(action_probs, dim=-1)

        action = torch.zeros(size=(len(state.batch_idxes), state.ope_num+1, state.station_num+3, 2),
                             dtype=torch.long, device=self.device)

        for i_batch in range(len(state.batch_idxes)):
            if action_index[i_batch] < (state.ope_num+1) * (state.station_num+3):
                action[i_batch, torch.div(action_index[i_batch], state.station_num+3, rounding_mode='trunc'),
                       action_index[i_batch] % (state.station_num+3), 0] = 1
            elif (state.ope_num+1) * (state.station_num+3) <= action_index[i_batch] < (
                    state.ope_num+1) * (state.station_num+3) * 2:
                action[i_batch, torch.div(action_index[i_batch] - (state.ope_num+1) * (state.station_num+3),
                                          state.station_num+3, rounding_mode='trunc'),
                       (action_index[i_batch] - (state.ope_num+1) * (state.station_num+3)) % (state.station_num+3), 1] = 1
            else:
                pass

        job_action_probs = self.get_job_prob(state, job_norm, action)
        dists_job = []
        job_actions = torch.zeros(size=(len(state.batch_idxes),), dtype=torch.long, device=self.device)
        for i_batch in range(len(state.batch_idxes)):
            dist_job = Categorical(job_action_probs[i_batch])
            dists_job.append(dist_job)
            if flag_sample:
                job_actions[i_batch] = dist_job.sample()
            else:
                job_actions[i_batch] = torch.argmax(job_action_probs[i_batch], dim=-1)

        if flag_train:
            memories.logprobs.append(dist.log_prob(action_index))
            memories.logprobs_job.append(torch.stack([dists_job[i_batch].log_prob(job_actions[i_batch])
                                                      for i_batch in range(len(state.batch_idxes))]))
            memories.action_envs.append(action_index)
            memories.action_job_envs.append(job_actions)

        return action, job_actions

    def evaluate(self, ope_ma_adj, ope_ma_adj_out, ope_buf_adj, ope_buf_adj_out, raw_opes, raw_mas, raw_buf,
                 raw_arc_ma_in, raw_arc_ma_out, raw_arc_buf_in, raw_arc_buf_out, raw_job, eligible, eligible_wait,
                 action_envs, action_job_envs):
        """
        Generate evaluate function
        """
        batch_idxes = torch.arange(0, raw_opes.size(0), dtype=torch.long, device=self.device)

        # L iterations of HGNN
        features = (raw_opes, raw_mas, raw_buf, raw_arc_ma_in, raw_arc_buf_in, raw_arc_ma_out, raw_arc_buf_out)
        adj = (ope_ma_adj, ope_ma_adj_out, ope_buf_adj, ope_buf_adj_out)

        for i_layer_hgnn in range(self.n_layers_hgnn):
            # Machine node embedding
            h_mas, h_buf = self.get_machines[i_layer_hgnn](adj, batch_idxes, features)
            features = (features[0], h_mas, h_buf, features[3], features[4], features[5], features[6])
            # Operation node embedding
            h_opes = self.get_operations[i_layer_hgnn](adj, batch_idxes, features)
            features = (h_opes, features[1], features[2], features[3], features[4], features[5], features[6])

        # Stacking and polling
        # Average pooling of the machine embedding node with shape (batch_size, out_size_ma)
        h_mas_pooled = torch.mean(h_mas, dim=-2)
        # Average pooling of the buffer embedding node with shape (batch_size, out_size_ma)
        h_buf_pooled = torch.mean(h_buf, dim=-2)
        # Average pooling of the machine and buffer embedding node with shape (batch_size, out_size_ma)
        h_mas_buf_pooled = torch.mean(torch.cat((h_mas, h_buf), dim=-2), dim=-2)
        # Average pooling of the operation embedding node with shape (batch_size, out_size_ope)
        h_opes_pooled = torch.mean(h_opes, dim=-2)

        # Average polling of the job embedding node with shape (batch_size, in_size_job)
        h_job_pooled = torch.zeros(size=(len(batch_idxes), self.in_size_job), dtype=torch.float, device=self.device)
        for i_idxes in range(len(batch_idxes)):
            h_job_pooled[i_idxes] = torch.mean(raw_job[i_idxes], dim=-2)

        # Structure the tensor with the same dimension
        h_opes_padding = h_opes.unsqueeze(-2).expand(-1, -1, h_mas.size(-2) + h_buf.size(-2), -1)
        h_mas_buf_padding = torch.cat((h_mas, h_buf), dim=-2).unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_opes_padding)
        h_mas_buf_pooled_padding = h_mas_buf_pooled[:, None, None, :].expand_as(h_mas_buf_padding)
        # h_mas_buf_pooled_padding = torch.cat(
        #     (h_mas_pooled, h_buf_pooled), dim=-1)[:, None, None, :].expand_as(h_mas_buf_padding)

        # Input of the actor network
        h_actions = torch.cat((
            h_opes_padding, h_mas_buf_padding, h_opes_pooled_padding, h_mas_buf_pooled_padding), dim=-1)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled, h_buf_pooled, h_job_pooled), dim=-1)

        # Get probability of actions with masking the ineligible actions
        scores = self.actor.forward(h_actions)
        scores[:, :, :, :2] = scores[:, :, :, :2].masked_fill(eligible == False, float('-inf'))
        # no_wait_action_scores = torch.max(scores[:, :, :, 0], scores[:, :, :, 1])
        # no_wait_action_scores = no_wait_action_scores.view(len(batch_idxes), -1)
        wait_action_scores = torch.mean(scores[:, :, :, 2], dim=(-1, -2)).unsqueeze(-1)
        wait_action_scores = wait_action_scores.masked_fill(eligible_wait == False, float('-inf'))
        # size: (len(batch_idxes), ope_num+1 * station_num+3 * 2 + 1)
        # action_scores = torch.cat((scores[:, :, :, 0], scores[:, :, :, 1], wait_action_scores), dim=-1)
        action_scores = torch.cat(
            (scores[:, :, :, 0].flatten(1), scores[:, :, :, 1].flatten(1), wait_action_scores), dim=-1)
        # action_scores = torch.cat((no_wait_action_scores, wait_action_scores), dim=-1)
        action_probs = F.softmax(action_scores, dim=-1)
        # print(action_probs.size())
        dist = Categorical(action_probs)
        actions_logprobs = dist.log_prob(action_envs)
        dist_entropy = dist.entropy()

        dist_job = []
        actions_job_logprobs = []
        dist_job_entropy = []

        for i_idxes in range(len(batch_idxes)):
            score_job = self.get_jobs.forward(raw_job[i_idxes]).squeeze(-1)
            action_job_prob = F.softmax(score_job, dim=-1)
            dist_job.append(Categorical(action_job_prob))
            actions_job_logprobs.append(dist_job[-1].log_prob(action_job_envs[i_idxes]))
            dist_job_entropy.append(dist_job[-1].entropy())

        actions_job_logprobs = torch.stack(actions_job_logprobs)
        dist_job_entropy = torch.stack(dist_job_entropy)

        # Calculate the value of the state
        state_value = self.critic.forward(h_pooled)    # size: (len(batch_idxes), 1)

        return actions_logprobs, actions_job_logprobs, state_value.squeeze(-1), dist_entropy, dist_job_entropy


class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras['lr']    # learning rate
        self.betas = train_paras['betas']    # Adam optimizer parameters
        self.gamma = train_paras['gamma']    # discount factor
        self.eps_clip = train_paras['eps_clip']    # clip parameter for PPO
        self.K_epochs = train_paras['K_epochs']    # update policy for K epochs
        self.A_coeff = train_paras['A_coeff']    # coefficient of policy loss
        self.V_coeff = train_paras['V_coeff']    # coefficient of value loss
        self.entropy_coeff = train_paras['entropy_coeff']    # coefficient of entropy loss
        self.num_envs = num_envs    # number of parallel environments
        self.device = torch.device(model_paras['device'])    # device

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = HGNNScheduler(model_paras).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MSELoss = nn.MSELoss()

    def update(self, memory, train_paras):
        device = self.device
        minibatch_size = train_paras['minibatch_size']
        # print(memory.logprobs)
        # print(memory.logprobs_job)

        # Flatten the data in memory (in the dimension of parallel instances and decision points)
        # old_ope_ma_adj = torch.stack(memory.ope_ma_adj).to(device)
        # old_ope_ma_adj_out = torch.stack(memory.ope_ma_adj_out).to(device)
        # old_ope_buf_adj = torch.stack(memory.ope_buf_adj).to(device)
        # old_ope_buf_adj_out = torch.stack(memory.ope_buf_adj_out).to(device)
        # old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_buf = torch.stack(memory.raw_buf, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_arc_ma_in = torch.stack(memory.raw_arc_ma_in, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_arc_ma_out = torch.stack(memory.raw_arc_ma_out, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_arc_buf_in = torch.stack(memory.raw_arc_buf_in, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_arc_buf_out = torch.stack(memory.raw_arc_buf_out, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_raw_job = copy.deepcopy(
        #     [item.to(device) for sublist in transpose_list_of_tensors(memory.raw_job) for item in sublist])
        # old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_eligible_wait = torch.stack(memory.eligible_wait, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_action_envs = torch.stack(memory.action_envs, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_action_job_envs = copy.deepcopy(
        #     [item.to(device) for sublist in transpose_list_of_tensors(memory.action_job_envs) for item in sublist])
        #
        # memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1).to(device)
        # memory_is_terminal = torch.stack(memory.is_terminals, dim=0).transpose(0, 1).to(device)
        #
        # old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1).to(device)
        # old_logprobs_job = torch.stack(memory.logprobs_job, dim=0).transpose(0, 1).flatten(0, 1).to(device)

        old_ope_ma_adj, old_ope_ma_adj_out, old_ope_buf_adj, old_ope_buf_adj_out, old_raw_opes, old_raw_mas, \
            old_raw_buf, old_raw_arc_ma_in, old_raw_arc_ma_out, old_raw_arc_buf_in, old_raw_arc_buf_out, old_raw_job, \
            old_eligible, old_eligible_wait, old_action_envs, old_action_job_envs, memory_rewards, memory_is_terminal, \
            old_logprobs, old_logprobs_job = memory_flatten(memory, device)

        # print(old_raw_job)

        # Estimate the rewards
        # rewards_envs = []
        # discounted_rewards = 0
        # for i_batch in range(self.num_envs):
        #     rewards = []
        #     discounted_reward = 0
        #     for reward, is_terminal in zip(reversed(memory_rewards[i_batch]), reversed(memory_is_terminal[i_batch])):
        #         if is_terminal:
        #             discounted_reward = 0
        #         discounted_reward = reward + (self.gamma * discounted_reward)
        #         rewards.insert(0, discounted_reward)
        #     discounted_rewards += discounted_reward
        #     rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        #     print("rewards: ", rewards)
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        #     rewards_envs.append(rewards)
        # rewards_envs = torch.cat(rewards_envs)
        # print("rewards_envs: ", rewards_envs.size())

        rewards_envs = []
        discounted_rewards = 0
        discounted_reward = 0
        # print("memory_rewards: ", memory_rewards)

        for reward, is_terminal in zip(reversed(memory_rewards), reversed(memory_is_terminal)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_envs.insert(0, discounted_reward)
            # discounted_rewards += discounted_reward
        rewards_envs = torch.tensor(rewards_envs, dtype=torch.float, device=device)
        # discounted_rewards = torch.sum(rewards_envs)

        terminal_i_idx = torch.nonzero(memory_is_terminal).squeeze() + 1
        terminal_i_idx = torch.cat((torch.tensor([0], device=device), terminal_i_idx), dim=0)

        for i_env_idx in range(self.num_envs):
            i_env_rewards = copy.deepcopy(rewards_envs[terminal_i_idx[i_env_idx]:terminal_i_idx[i_env_idx+1]])
            discounted_rewards += i_env_rewards[0]
            i_env_rewards = (i_env_rewards - i_env_rewards.mean()) / (i_env_rewards.std() + 1e-5)
            rewards_envs[terminal_i_idx[i_env_idx]:terminal_i_idx[i_env_idx+1]] = i_env_rewards

        loss_epochs = 0
        full_batch_size = old_raw_opes.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)    # 应该是这里被恰好整除了
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for i_minibatch in range(num_complete_minibatches+1):
                if i_minibatch < num_complete_minibatches:
                    start_idx = i_minibatch * minibatch_size
                    end_idx = (i_minibatch + 1) * minibatch_size
                else:
                    start_idx = i_minibatch * minibatch_size
                    end_idx = full_batch_size

                if start_idx == end_idx:
                    break

                # print(len(old_raw_job[start_idx:end_idx]))
                # print(old_action_envs)
                logprobs, logprobs_job, state_values, dist_entropy, dist_job_entropy = self.policy.evaluate(
                    old_ope_ma_adj, old_ope_ma_adj_out, old_ope_buf_adj, old_ope_buf_adj_out,
                    old_raw_opes[start_idx:end_idx, :, :], old_raw_mas[start_idx:end_idx, :, :],
                    old_raw_buf[start_idx:end_idx, :, :], old_raw_arc_ma_in[start_idx:end_idx, :, :, :],
                    old_raw_arc_ma_out[start_idx:end_idx, :, :, :], old_raw_arc_buf_in[start_idx:end_idx, :, :, :],
                    old_raw_arc_buf_out[start_idx:end_idx, :, :, :], old_raw_job[start_idx:end_idx],
                    old_eligible[start_idx:end_idx, :, :, :], old_eligible_wait[start_idx:end_idx],
                    old_action_envs[start_idx:end_idx], old_action_job_envs[start_idx:end_idx])

                ratios_arc = torch.exp(logprobs - old_logprobs[start_idx:end_idx].detach())
                ratios_job = torch.exp(logprobs_job - old_logprobs_job[start_idx:end_idx].detach())
                advantages = rewards_envs[start_idx:end_idx].unsqueeze(-1) - state_values.detach()
                # ratios = (ratios_arc + ratios_job) / 2
                # surr1 = ratios * advantages
                # surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                # loss = - self.A_coeff * torch.min(surr1, surr2) + self.V_coeff * self.MSELoss(
                #     state_values, rewards_envs[start_idx:end_idx]) \
                #     - self.entropy_coeff * (dist_entropy + dist_job_entropy)
                # loss_epochs += loss.mean().detach()

                surr1_arc = ratios_arc * advantages
                surr2_arc = torch.clamp(ratios_arc, 1-self.eps_clip, 1+self.eps_clip) * advantages
                surr1_job = ratios_job * advantages
                surr2_job = torch.clamp(ratios_job, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = - self.A_coeff * torch.min(surr1_arc, surr2_arc) \
                    - self.A_coeff * torch.min(surr1_job, surr2_job) \
                    + self.V_coeff * self.MSELoss(state_values, rewards_envs[start_idx:end_idx]) \
                    - self.entropy_coeff * (dist_entropy + dist_job_entropy)
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, discounted_rewards.item() / (
                self.num_envs * train_paras["update_timestep"])











