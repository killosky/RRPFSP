import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F


class GATedge(nn.Module):
    """
    Machine node embedding
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 device=torch.device("cuda")):
        """
        :param in_feats: tuple, input dimension of (operation node, machine node, arc)
        :param out_feats: output dimension of the embedded machine node features
        """
        super(GATedge, self).__init__()
        self.device = device
        self._in_ope_feats = in_feats[0]
        self._in_mas_feats = in_feats[1]
        self._in_arc_feats = in_feats[2]
        self._out_feats = out_feats

        # Generate linear layer for operation node, machine node and acr respectively
        self.fc_ope = nn.Linear(self._in_ope_feats, out_feats * num_head, bias=False)
        self.fc_mas = nn.Linear(self._in_mas_feats, out_feats * num_head, bias=False)
        self.fc_buf = nn.Linear(self._in_mas_feats, out_feats * num_head, bias=False)
        self.fc_arc_in = nn.Linear(self._in_arc_feats, out_feats * num_head, bias=False)
        self.fc_arc_out = nn.Linear(self._in_arc_feats, out_feats * num_head, bias=False)

        # Initialize attention parameters
        self.attn_ope = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_mas = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_arc = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def reset_parameters(self):
        """
        Reinitialize model parameters
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_ope.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_mas.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_buf.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_arc_in.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_arc_out.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_ope, gain=gain)
        nn.init.xavier_normal_(self.attn_mas, gain=gain)
        nn.init.xavier_normal_(self.attn_arc, gain=gain)

    def forward(self, adj, batch_idxes, feat):
        """
        forward propagation
        :param adj: (ope_ma_adj, ope_ma_adj_out, ope_buf_adj, ope_buf_adj_out)
        :param feat: (feat_opes_batch, feat_mas_batch, feat_buf_batch, feat_arc_ma_in_batch, feat_arc_buf_in_batch,
                        feat_arc_ma_out_batch, feat_arc_buf_out_batch)
        """
        # dropout for operation node, machine node and arc respectively
        h_opes = self.feat_drop(feat[0])
        h_mas = self.feat_drop(feat[1])
        h_buf = self.feat_drop(feat[2])
        h_arc_ma_in = self.feat_drop(feat[3])
        h_arc_buf_in = self.feat_drop(feat[4])
        h_arc_ma_out = self.feat_drop(feat[5])
        h_arc_buf_out = self.feat_drop(feat[6])

        feat_ope = self.fc_ope(h_opes)
        feat_mas = self.fc_mas(h_mas)
        feat_buf = self.fc_buf(h_buf)
        feat_arc_ma_in = self.fc_arc_in(h_arc_ma_in)
        feat_arc_buf_in = self.fc_arc_in(h_arc_buf_in)
        feat_arc_ma_out = self.fc_arc_out(h_arc_ma_out)
        feat_arc_buf_out = self.fc_arc_out(h_arc_buf_out)

        # Calculate attention coefficient
        e_ope = (feat_ope * self.attn_ope).sum(dim=-1).unsqueeze(-1)
        e_mas = (feat_mas * self.attn_mas).sum(dim=-1).unsqueeze(-1)
        e_buf = (feat_buf * self.attn_mas).sum(dim=-1).unsqueeze(-1)
        e_arc_ma_in = (feat_arc_ma_in * self.attn_arc).sum(dim=-1).unsqueeze(-1)
        e_arc_buf_in = (feat_arc_buf_in * self.attn_arc).sum(dim=-1).unsqueeze(-1)
        e_arc_ma_out = (feat_arc_ma_out * self.attn_arc).sum(dim=-1).unsqueeze(-1)
        e_arc_buf_out = (feat_arc_buf_out * self.attn_arc).sum(dim=-1).unsqueeze(-1)

        # Calculate embedding feature for machine station node
        e_ope_add_e_arc_ma_in = adj[0].unsqueeze(0).unsqueeze(-1) * e_ope.unsqueeze(-2) + e_arc_ma_in
        e_ope_add_e_arc_ma_out = adj[1].unsqueeze(0).unsqueeze(-1) * e_ope.unsqueeze(-2) + e_arc_ma_out
        a_in = e_ope_add_e_arc_ma_in + adj[0].unsqueeze(0).unsqueeze(-1) * e_mas.unsqueeze(-3)
        a_out = e_ope_add_e_arc_ma_out + adj[1].unsqueeze(0).unsqueeze(-1) * e_mas.unsqueeze(-3)
        eik_in = self.leaky_relu(a_in)
        eik_out = self.leaky_relu(a_out)
        ekk = self.leaky_relu(e_mas + e_mas)

        # Normalize attention scores using softmax
        mask_in = torch.cat((adj[0].unsqueeze(-1) == 1, torch.full(
            size=(1, adj[0].size(1), 1), dtype=torch.bool, fill_value=True, device=self.device)), dim=-3) \
            .unsqueeze(0).expand(batch_idxes.size(0), -1, -1, -1)
        mask_out = torch.cat((adj[1].unsqueeze(-1) == 1, torch.full(
            size=(1, adj[1].size(1), 1), dtype=torch.bool, fill_value=True, device=self.device)), dim=-3) \
            .unsqueeze(0).expand(batch_idxes.size(0), -1, -1, -1)
        e_in = torch.cat((eik_in, ekk.unsqueeze(-3)), dim=-3)
        e_out = torch.cat((eik_out, ekk.unsqueeze(-3)), dim=-3)
        e_in[~mask_in] = float('-inf')
        e_out[~mask_out] = float('-inf')
        alpha_in = F.softmax(e_in.squeeze(-1), dim=-2)
        alpha_ik_in = alpha_in[..., :-1, :]
        alpha_kk_in = alpha_in[..., -1, :].unsqueeze(-2)
        alpha_out = F.softmax(e_out.squeeze(-1), dim=-2)
        alpha_ik_out = alpha_out[..., :-1, :]
        alpha_kk_out = alpha_out[..., -1, :].unsqueeze(-2)

        Wmu_ik_in = torch.sum((feat_arc_ma_in + feat_ope.unsqueeze(-2)) * alpha_ik_in.unsqueeze(-1), dim=-3)
        Wmu_ik_out = torch.sum(feat_arc_ma_out + feat_ope.unsqueeze(-2) * alpha_ik_out.unsqueeze(-1), dim=-3)
        Wmu_kk = feat_mas * (alpha_kk_in + alpha_kk_out).squeeze(-2).unsqueeze(-1)
        nu_k_prime = torch.sigmoid(Wmu_ik_in + Wmu_ik_out + Wmu_kk)

        # Calculate embedding feature for buffer node
        e_ope_add_e_arc_buf_in = adj[2].unsqueeze(0).unsqueeze(-1) * e_ope.unsqueeze(-2) + e_arc_buf_in
        e_ope_add_e_arc_buf_out = adj[3].unsqueeze(0).unsqueeze(-1) * e_ope.unsqueeze(-2) + e_arc_buf_out
        a_buf_in = e_ope_add_e_arc_buf_in + adj[2].unsqueeze(0).unsqueeze(-1) * e_buf.unsqueeze(-3)
        a_buf_out = e_ope_add_e_arc_buf_out + adj[3].unsqueeze(0).unsqueeze(-1) * e_buf.unsqueeze(-3)
        eik_buf_in = self.leaky_relu(a_buf_in)
        eik_buf_out = self.leaky_relu(a_buf_out)
        ekk_buf = self.leaky_relu(e_buf + e_buf)

        mask_buf_in = torch.cat((adj[2].unsqueeze(-1) == 1, torch.full(
            size=(1, adj[2].size(1), 1), dtype=torch.bool, fill_value=True, device=self.device)), dim=-3) \
            .unsqueeze(0).expand(batch_idxes.size(0), -1, -1, -1)
        mask_buf_out = torch.cat((adj[3].unsqueeze(-1) == 1, torch.full(
            size=(1, adj[3].size(1), 1), dtype=torch.bool, fill_value=True, device=self.device)), dim=-3) \
            .unsqueeze(0).expand(batch_idxes.size(0), -1, -1, -1)
        e_buf_in = torch.cat((eik_buf_in, ekk_buf.unsqueeze(-3)), dim=-3)
        e_buf_out = torch.cat((eik_buf_out, ekk_buf.unsqueeze(-3)), dim=-3)
        e_buf_in[~mask_buf_in] = float('-inf')
        e_buf_out[~mask_buf_out] = float('-inf')
        alpha_buf_in = F.softmax(e_buf_in.squeeze(-1), dim=-2)
        alpha_ik_buf_in = alpha_buf_in[..., :-1, :]
        alpha_kk_buf_in = alpha_buf_in[..., -1, :].unsqueeze(-2)
        alpha_buf_out = F.softmax(e_buf_out.squeeze(-1), dim=-2)
        alpha_ik_buf_out = alpha_buf_out[..., :-1, :]
        alpha_kk_buf_out = alpha_buf_out[..., -1, :].unsqueeze(-2)

        Wmu_ik_buf_in = torch.sum((feat_arc_buf_in + feat_ope.unsqueeze(-2)) * alpha_ik_buf_in.unsqueeze(-1), dim=-3)
        Wmu_ik_buf_out = torch.sum(feat_arc_buf_out + feat_ope.unsqueeze(-2) * alpha_ik_buf_out.unsqueeze(-1), dim=-3)
        Wmu_kk_buf = feat_buf * (alpha_kk_buf_in + alpha_kk_buf_out).squeeze(-2).unsqueeze(-1)
        nu_k_buf_prime = torch.sigmoid(Wmu_ik_buf_in + Wmu_ik_buf_out + Wmu_kk_buf)

        return nu_k_prime, nu_k_buf_prime


class MLPsim(nn.Module):
    """
    Part of operation node embedding
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 device=torch.device("cuda")):
        """
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        """
        super(MLPsim, self).__init__()
        self.device = device

        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats * self._num_heads)
        )

    def forward(self, feat, adj):
        """
        forward propagation
        """
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        b = torch.sum(a, dim=-2)
        c = self.project(b)

        return c