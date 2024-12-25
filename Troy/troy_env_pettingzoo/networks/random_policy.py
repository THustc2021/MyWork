from typing import Any, cast

import numpy as np
import torch

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy


class RandomPolicy(BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute the random action over the given batch data.

        The input should contain a mask in batch.obs, with "True" to be
        available and "False" to be unavailable. For example,
        ``batch.obs.mask == np.array([[False, True, False]])`` means with batch
        size 1, action "1" is available but action "0" and "2" are unavailable.

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the random action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        # 为每个最后一个维度创建一个随机变量
        commands = []
        for command_idx in range(batch.obs.shape[2]):
            command = np.random.rand(*batch.obs.shape[:-1], self.action_space.high[0, command_idx]+1)
            command = command.argmax(axis=-1)
            commands.append(command)
        logits = np.stack(commands, axis=-1)
        # 动作掩码
        mask = batch.obs.mask  # type: ignore
        logits[~mask.astype(bool)] = 0
        result = Batch(act=logits)
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}
