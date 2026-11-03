"""LIF source with a stim-encoded reward channel.

The SDK gives experiment code no handle on the simulator subprocess, so
``LIFDataSource.apply_reward`` is unreachable from a closed loop. This subclass
treats stims on two reserved channels as reward signals instead of current:

* ``REWARD_CHANNEL`` (1)  -> ``apply_reward(+1)``
* ``PUNISH_CHANNEL`` (2)  -> ``apply_reward(-1)``

All other stims are passed through unchanged. Simulator-only convention: on a
CL1 these are ordinary stims.
"""
from __future__ import annotations

from .lif_data_source import LIFDataSource

REWARD_CHANNEL = 1
PUNISH_CHANNEL = 2


class RewardLIFDataSource(LIFDataSource):
    def on_stims(self, stims):
        passthrough = []
        for stim in stims:
            if stim.channel == REWARD_CHANNEL:
                self.apply_reward(+1.0)
            elif stim.channel == PUNISH_CHANNEL:
                self.apply_reward(-1.0)
            else:
                passthrough.append(stim)
        super().on_stims(passthrough)


def make_reward_lif_source(**kwargs) -> RewardLIFDataSource:
    """Factory for ``cl.sim.set_simulator_data_source``."""
    return RewardLIFDataSource(**kwargs)
