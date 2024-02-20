

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from graph.hgnn import GATedge, MLPsim
from mlp import MLPActor, MLPCritic


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


def get_normalized(raw_opes, raw_mas, raw_buf, raw_job, batch_idxes):
    """
    Normalized time variable in feats, including operations, machines and arcs feats.
    """
    mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
    std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
    mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
    std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
    mean_buf = torch.mean(raw_buf, dim=-2, keepdim=True)
    std_buf = torch.std(raw_buf, dim=-2, keepdim=True)

    feat_job_normalized = []
    for i_idxes in range(len(batch_idxes)):
        mean_jobs = torch.mean(raw_job[i_idxes], dim=-2, keepdim=True)
        std_jobs = torch.std(raw_job[i_idxes], dim=-2, keepdim=True)
        feat_job_normalized.append((raw_job[i_idxes] - mean_jobs) / (std_jobs + 1e-5))

    return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
            (raw_buf - mean_buf) / (std_buf + 1e-5), feat_job_normalized)


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

        self.n_hidden_actor = model_paras['n_hidden_actor']  # Hidden dimension of the actor network
        self.n_hidden_critic = model_paras['n_hidden_critic']  # Hidden dimension of the critic network
        self.n_layers_actor = model_paras['n_layers_actor']  # Number of layers of the actor network
        self.n_layers_critic = model_paras['n_layers_critic']  # Number of layers of the critic network

        self.n_layers_hgnn = model_paras['n_layers_hgnn']  # Number of layers of the HGNN GATedge

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

    def forward(self):
        raise NotImplementedError

    def get_arc_prob(self):
        pass
