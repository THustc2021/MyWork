import random
from typing import Sequence, Any, no_type_check, cast

import torch
import numpy as np
import tianshou as ts
from tianshou.data import Batch
from torch import nn
from tianshou.utils.net.common import MLP

def sawtoothwave(x, period=2 * torch.pi):
    return (2 * (x / period - torch.floor(0.5 + x / period)) + 1) * 0.5


class PreProcessNet(nn.Module):
    def __init__(self, hidden_dim, observation_space, device="cpu"):
        super(PreProcessNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        self.time_low = observation_space["obs_time"].low
        self.time_high = observation_space["obs_time"].high
        self.time_shape = np.prod(observation_space["obs_time"].shape)
        self.self_low = observation_space["obs_self"].low
        self.self_high = observation_space["obs_self"].high
        self.self_shape = np.prod(observation_space["obs_self"].shape)
        self.enemy_low = observation_space["obs_enemy"].low
        self.enemy_high = observation_space["obs_enemy"].high
        self.enemy_shape = np.prod(observation_space["obs_enemy"].shape)

        # 有必要将上次的状态传过来？
        self.state_process = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()    # 遗忘
        )

        self.model = nn.Sequential(
            nn.Linear(self.time_shape + self.self_shape + self.enemy_shape, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        nums, dims = observation_space["obs_self"].shape
        self.self_embed = nn.Linear(dims, dims)
        self.self_embed_pos = torch.linspace(start=0.0, end=nums-1, steps=nums)[None]
        nume, dime = observation_space["obs_enemy"].shape
        self.enemy_embed = nn.Linear(dime, dime)
        self.enemy_embed_pos = torch.linspace(start=0.0, end=nume-1, steps=nume)[None]

    def with_pos_embed(self, t, pos):
        return t + pos[..., None].repeat(t.shape[0], 1, 1).to(t)

    def forward(self, obs, state=None):
        # obs_time: 1
        # obs_self: (envn, num, a)
        # obs_enemy: (envn, enum, b)
        # 输入归一化到[0, 1]
        # return: hidden_dim
        time_norm = (obs.obs_time - self.time_low) / (self.time_high - self.time_low)
        self_norm = (obs.obs_self - self.self_low) / (self.self_high - self.self_low)
        enemy_norm = (obs.obs_enemy - self.enemy_low) / (self.enemy_high - self.enemy_low)
        time_norm_t = torch.tensor(time_norm, dtype=torch.float)[:, None].to(self.device)

        # 附上位置编码
        self_norm_t = torch.tensor(self_norm, dtype=torch.float).to(self.device)
        self_norm_t = self.with_pos_embed(self.self_embed(self_norm_t), self.self_embed_pos).flatten(1)
        enemy_norm_t = torch.tensor(enemy_norm, dtype=torch.float).to(self.device)
        enemy_norm_t = self.with_pos_embed(self.enemy_embed(enemy_norm_t), self.enemy_embed_pos).flatten(1)

        obs_t = torch.cat([time_norm_t, self_norm_t, enemy_norm_t], dim=1).to(self.device)
        logits = self.model(obs_t)
        if state != None and (isinstance(state, Batch) and not state.is_empty()):
            logits = logits + self.state_process(state.state)
        return logits, Batch(state=logits)

class Actor(nn.Module):
    """Simple actor network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param restrict_range: a int for the relative range
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param max_action: the scale for the final action logits. Default to
        1.
    :param preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        prepreprocess_net: PreProcessNet,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_actions: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.prepreprocess = prepreprocess_net
        self.preprocess = preprocess_net

        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        input_dim = cast(int, input_dim)
        self.last = MLP(    # 用原生MLP即可
            input_dim,
            self.output_dim,
            hidden_sizes,
            device=self.device,
        )
        self.max_actions = max_actions

        # class head
        self.cls_head = nn.ModuleList()
        for i in range(len(self.max_actions)):
            self.cls_head.append(
                nn.Linear(self.output_dim, action_shape[0]*(self.max_actions[i]+1)) #  + 1 类匹配总体数量
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ):
        """Mapping: obs -> logits -> action."""

        if info is None:
            info = {}
        if not hasattr(obs, "mask"):
            obs = obs.obs   # 我们直接认为，没有mask就是存在嵌套（意味着来自replaybuffer？）
        # 区分obs和mask
        mask = obs.mask

        # 预处理
        obs_pre, st = self.prepreprocess(obs, state)
        # 得到处理后的特征，进一步计算
        logits, hidden = self.preprocess(obs_pre, st)    # 默认的preprocess不会对state作任何处理，也是原样返回
        logits = self.last(logits)

        # 最终预测
        n, nn, _ = mask.shape
        ts = []
        for mi in range(len(self.cls_head)):
            m = self.cls_head[mi]
            t = m(logits).view(n, nn, -1)
            ts.append(torch.softmax(t, dim=-1))

        return torch.cat(ts, dim=-1).to(logits), hidden

class Critic(nn.Module):
    """Simple critic network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        prepreprocess_net: PreProcessNet,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
        linear_layer: type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.prepreprocess = prepreprocess_net
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        if info is None:
            info = {}
        if not hasattr(obs, "mask"):
            obs = obs.obs   # 我们直接认为，没有mask就是存在嵌套
        # 预处理
        obs, st = self.prepreprocess(obs)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        logits, hidden = self.preprocess(obs, st)
        return self.last(logits)
