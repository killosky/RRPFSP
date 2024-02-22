

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from graph.hgnn import GATedge, MLPsim
from mlp import MLPActor, MLPCritic, MLPJob


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

        for i_feat in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(
                self.in_sizes_ope[i_feat], self.out_size_ope, self.hidden_size_ope, self.num_head).to(self.device))
        self.project = nn.Sequential(
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
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
        for i_feat in range(len(self.in_sizes_ope)):
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
        mean_jobs = torch.mean(raw_job[i_idxes], dim=-2, keepdim=True)
        std_jobs = torch.std(raw_job[i_idxes], dim=-2, keepdim=True)
        feat_job_normalized.append((raw_job[i_idxes] - mean_jobs) / (std_jobs + 1e-5))

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

        self.actor_dim = model_paras['actor_dim']    # Input dimension of the actor network (NOT DEFINED IN THE JSON FILE)
        self.critic_dim = model_paras['critic_dim']    # Input dimension of the critic network (NOT DEFINED IN THE JSON FILE)
        self.job_section_dim = model_paras['job_section_dim']    # Input dimension of the job section network (NOT DEFINED IN THE JSON FILE)

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
                                         attn_drop=self.dropout).to(self.device))
        for i_layer_hgnn in range(self.n_layers_hgnn - 1):
            self.get_machines.append(GATedge(in_feats=(self.out_size_ope, self.out_size_ma, self.in_size_arc),
                                             out_feats=self.out_size_ma, num_head=self.num_head[0],
                                             feat_drop=self.dropout, attn_drop=self.dropout).to(self.device))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs(in_sizes_ope_embed=[
            self.in_size_ope, self.in_size_ope, self.in_size_ma, self.in_size_ma, self.in_size_ma, self.in_size_ope],
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

    def get_arc_prob(self, state):
        """
        Get the probability of each arc in decision-making
        """
        # Uncompleted instances
        batch_idxes = state.batch_idxes

        # Raw features
        raw_opes = state.feat_opes_batch[batch_idxes]
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
        h_mas_buf_pooled_padding = torch.cat(
            (h_mas_pooled, h_buf_pooled), dim=-1)[:, None, None, :].expand_as(h_mas_buf_padding)

        # Input of the actor network
        h_actions = torch.cat((
            h_opes_padding, h_mas_buf_pooled_padding, h_opes_pooled_padding, h_mas_buf_pooled_padding), dim=-1)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled, h_buf_pooled, h_job_pooled), dim=-1)

        # Get probability of actions with masking the ineligible actions
        scores = self.actor.forward(h_actions)
        scores[:, :, :, :2] = scores[:, :, :, :2].masked_fill(eligible == False, float('-inf'))
        # no_wait_action_scores = torch.max(scores[:, :, :, 0], scores[:, :, :, 1])
        # no_wait_action_scores = no_wait_action_scores.view(len(batch_idxes), -1)
        wait_action_scores = torch.mean(scores[:, :, :, 2], dim=(-1, -2)).unsqueeze(-1)
        wait_action_scores = wait_action_scores.masked_fill(eligible_wait == False, float('-inf'))
        # size: (len(batch_idxes), ope_num+1 * station_num+3 * 2 + 1)
        action_scores = torch.cat((scores[:, :, :, 0], scores[:, :, :, 1], wait_action_scores), dim=-1)
        # action_scores = torch.cat((no_wait_action_scores, wait_action_scores), dim=-1)
        action_probs = F.softmax(action_scores, dim=-1)

        return action_probs, h_pooled

    def get_job_prob(self, state, arc_action):
        """
        Get the job idx corresponding to each arc selection in each batch
        :param state: state of the environment
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

                score_job = self.get_jobs.forward(state.feat_job_batch[state.batch_idxes[i_idxes]])
                score_job = score_job.masked_fill(selection_job_mask == 0, float('-inf'))
                action_job_prob = F.softmax(score_job, dim=-1)

            else:
                action_job_prob = torch.zeros(size=(state.job_num[i_idxes],), dtype=torch.float, device=self.device)

            action_job_probs.append(action_job_prob)

        return action_job_probs

    def act(self, state, memories, done, flag_sample=True, flag_train=True):
        """
        Get actions of the env
        """
        action_probs, _ = self.get_arc_prob(state)
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
                       action_index[i_batch] % (state.ope_num+1), 0] = 1
            elif (state.ope_num+1) * (state.station_num+3) <= action_index[i_batch] < (
                    state.ope_num+1) * (state.station_num+3) * 2:
                action[i_batch, torch.div(action_index[i_batch] - (state.ope_num+1) * (state.station_num+3),
                                          state.station_num+3, rounding_mode='trunc'),
                       (action_index[i_batch] - (state.ope_num+1) * (state.station_num+3)) % (state.ope_num+1), 1] = 1
            else:
                pass

        job_action_probs = self.get_job_prob(state, action)
        job_actions = torch.zeros(size=(len(state.batch_idxes),), dtype=torch.long, device=self.device)
        for i_batch in range(len(state.batch_idxes)):
            if torch.nonzero(job_action_probs[i_batch]).size(0) > 0:
                dist_job = Categorical(job_action_probs[i_batch])
                if flag_sample:
                    job_actions[i_batch] = dist_job.sample()
                else:
                    job_actions[i_batch] = torch.argmax(job_action_probs[i_batch], dim=-1)

        # if flag_train:
        #     memories.states.append(copy.deepcopy(state))
        #     memories.actions.append(action)
        #     memories.job_actions.append(job_actions)
        #     memories.dones.append(done)

        return action, job_actions

    def evaluate(self, ope_ma_adj, ope_ma_adj_out, ope_buf_adj, ope_buf_adj_out, raw_opes, raw_mas, raw_buf,
                 raw_arc_ma_in, raw_arc_ma_out, raw_arc_buf_in, raw_arc_buf_out, raw_job, eligible, eligible_wait,
                 action_envs, action_job_envs):
        """
        Generate evaluate function
        """
        batch_idxes = torch.arange(0, raw_opes.size(0), dtype=torch.long, device=self.device)

        # Normalized input features
        opes_norm, mas_norm, buf_norm, arc_ma_in_norm, arc_ma_out_norm, arc_buf_in_norm, arc_buf_out_norm, job_norm = \
            get_normalized(raw_opes, raw_mas, raw_buf, raw_arc_ma_in, raw_arc_ma_out, raw_arc_buf_in, raw_arc_buf_out,
                           raw_job, batch_idxes)

        # L iterations of HGNN
        features = (opes_norm, mas_norm, buf_norm, arc_ma_in_norm, arc_buf_in_norm, arc_ma_out_norm, arc_buf_out_norm)
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
        # Average pooling of the operation embedding node with shape (batch_size, out_size_ope)
        h_opes_pooled = torch.mean(h_opes, dim=-2)

        # Average polling of the job embedding node with shape (batch_size, in_size_job)
        h_job_pooled = torch.zeros(size=(len(batch_idxes), self.in_size_job), dtype=torch.float, device=self.device)
        for i_idxes in range(len(batch_idxes)):
            h_job_pooled[i_idxes] = torch.mean(job_norm[i_idxes], dim=-2)

        # Structure the tensor with the same dimension
        h_opes_padding = h_opes.unsqueeze(-2).expand(-1, -1, h_mas.size(-2) + h_buf.size(-2), -1)
        h_mas_buf_padding = torch.cat((h_mas, h_buf), dim=-2).unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_opes_padding)
        h_mas_buf_pooled_padding = torch.cat(
            (h_mas_pooled, h_buf_pooled), dim=-1)[:, None, None, :].expand_as(h_mas_buf_padding)

        # Input of the actor network
        h_actions = torch.cat((
            h_opes_padding, h_mas_buf_pooled_padding, h_opes_pooled_padding, h_mas_buf_pooled_padding), dim=-1)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled, h_buf_pooled, h_job_pooled), dim=-1)

        # Get probability of actions with masking the ineligible actions
        scores = self.actor.forward(h_actions)
        scores[:, :, :, :2] = scores[:, :, :, :2].masked_fill(eligible == False, float('-inf'))
        # no_wait_action_scores = torch.max(scores[:, :, :, 0], scores[:, :, :, 1])
        # no_wait_action_scores = no_wait_action_scores.view(len(batch_idxes), -1)
        wait_action_scores = torch.mean(scores[:, :, :, 2], dim=(-1, -2)).unsqueeze(-1)
        wait_action_scores = wait_action_scores.masked_fill(eligible_wait == False, float('-inf'))
        # size: (len(batch_idxes), ope_num+1 * station_num+3 * 2 + 1)
        action_scores = torch.cat((scores[:, :, :, 0], scores[:, :, :, 1], wait_action_scores), dim=-1)
        # action_scores = torch.cat((no_wait_action_scores, wait_action_scores), dim=-1)
        action_probs = F.softmax(action_scores, dim=-1)
        dist = Categorical(action_probs)
        actions_logprobs = dist.log_prob(action_envs)
        dist_entropy = dist.entropy()

        action_job_probs_no_mask = []
        dist_job = []
        actions_job_logprobs = []
        dist_job_entropy = []

        for i_idxes in range(len(batch_idxes)):
            score_job = self.get_jobs.forward(raw_job[batch_idxes[i_idxes]])
            action_job_prob = F.softmax(score_job, dim=-1)
            action_job_probs_no_mask.append(action_job_prob)
            dist_job.append(Categorical(action_job_prob))
            actions_job_logprobs.append(dist_job[i_idxes].log_prob(action_job_envs[i_idxes]))
            dist_job_entropy.append(dist_job[i_idxes].entropy())

        # Calculate the value of the state
        state_value = self.critic.forward(h_pooled)    # size: (len(batch_idxes), 1)

        return actions_logprobs, actions_job_logprobs, state_value, dist_entropy, dist_job_entropy













