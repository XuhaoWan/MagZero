# Examples

Minimal training example:

```bash
magzero-train       --config configs/default.yaml       --data-folder /abs/path/to/heterodata       --rf-data /abs/path/to/graph_data.csv       --no-wandb
```

Minimal inference example:

```bash
magzero-infer       --checkpoint /abs/path/to/best.ckpt       --config configs/default.yaml       --data-folder /abs/path/to/heterodata       --rf-data /abs/path/to/graph_data.csv       --output-csv outputs/results.csv
```
