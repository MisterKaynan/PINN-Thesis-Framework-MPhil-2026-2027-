# Model checkpoints

Supplementary material for CSCD 618 / DSCD 604. Each `.pth` holds the trained
weights behind a specific claim in the report.

| File | What it is | Report reference |
|---|---|---|
| `validation_split.pth` | Millevoi Case 4, split schedule | Table 1 (replication) |
| `validation_joint.pth` | Millevoi Case 4, joint schedule | Table 1 (replication) |
| `ghana_split_daily.pth` | Ghana, dense daily, origin 220 | Table 2, headline MASE |
| `ghana_joint_daily.pth` | Ghana, dense daily, origin 220 | Table 2 |
| `ghana_split_mask.pth` | Ghana, authentic reporting mask | Table 2 |
| `rt_split_90day.pth` | 90-day window | Table 3, R_t comparison |

## Loading

```python
import sys; sys.path.insert(0, "src")
import torch, numpy as np
from pinn import ReducedSIRPINN

model, meta = ReducedSIRPINN.load("results/checkpoints/ghana_split_daily.pth")
print(meta)                       # source, mode, seed, training time

t = torch.tensor(np.arange(220) / 220, dtype=torch.float32).reshape(-1, 1)
I_pred, Rt = model.predict(t)     # I_pred in case units, Rt dimensionless
```

## What is inside

Each checkpoint stores both network state dicts plus the constants needed to
use them:

- `I_net`, `R_net` — weights for the infection and reproduction-number networks
- `config` — the full `Config` (architecture, epochs, collocation points, seed)
- `tf`, `scale_I` — time and state scaling factors
- `history` — loss trace
- `meta` — data source, schedule, seed, training seconds, date range

`tf` and `scale_I` matter. The networks take scaled time on [0, 1] and return
scaled state, so weights alone cannot be mapped back to case counts. Anything
that reads these files must apply the same scaling, which is why it travels
with the weights rather than living in a script.

## Regenerating

```bash
PYTHONPATH=src python src/make_checkpoints.py --workers 12
PYTHONPATH=src python src/make_checkpoints.py --workers 12 --skip-joint   # ~5 min
```

Roughly 35 minutes for the full set on 12 vCPUs; the two joint checkpoints are
most of that. Total size is under 1 MB.
