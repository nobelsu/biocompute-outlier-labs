# Neural Catcher — design

Hackathon demo (1 h, solo, simulator only, judged on wow factor): a simulated
organoid plays a catcher game and visibly improves.

## Loop (20 Hz, `experiments/catcher.py`)
- **Encode:** relative orb position `dx = orb_x - paddle_x` is binned into 6
  sensor channels (25–30, row 3); the bin's channel is stimulated every tick.
- **Decode:** spike counts on all 64 channels → `score = w · counts` →
  p(right) = sigmoid(score); action sampled (left/right).
- **Learn (readout):** REINFORCE eligibility trace `e += (a - E[a]) * counts`;
  on catch/miss `w += lr * r * e`. Guarantees improvement.
- **Learn (tissue, bonus):** LIF `plasticity=True`; a reward bridge turns
  stims on channel 1 (+1) / channel 2 (−1) into `apply_reward()`.

## Reward bridge (`src/organoid_simulator/reward_lif.py`)
`RewardLIFDataSource(LIFDataSource)` intercepts stims on reward channels in
`on_stims`, calls `apply_reward`, forwards the rest. Simulator-only convention.

## Screen (pygame-ce)
Game on the left; right panel: live 8×8 spike heatmap, learned readout map
(left vs right channels), rolling catch-rate curve, score. Keys: SPACE toggle
learning, R reset readout, ESC quit. `--headless` prints catch rate per block
for accelerated verification.

## Honesty note
Most of the learning is the Python readout; LIF plasticity is a bonus effect.
Simulator results are not evidence about CL1 tissue.
