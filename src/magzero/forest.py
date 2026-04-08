from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch import nn


class ForestWrapper(nn.Module):
    """Random-forest helper used to inject tabular physical priors.

    The random forest is trained once during initialization from a CSV file.
    During the forward pass, the module returns the forest prediction as a tensor.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        csv_path: str | Path = "graph_data.csv",
        n_estimators: int = 100,
        random_state: int = 66,
    ):
        super().__init__()
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.proj = nn.Linear(input_dim, output_dim)
        self.csv_path = str(csv_path)

csv_path = Path(self.csv_path)
if not csv_path.exists():
    raise FileNotFoundError(
        f"Random-forest feature CSV not found: {csv_path}. "
        "Provide --rf-data or update rf_csv_path in the model configuration."
    )

data = pd.read_csv(csv_path)
        x = data.iloc[:, :input_dim].values
        y = data.iloc[:, input_dim].values

        x_train, _, y_train, _ = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=random_state,
        )
        self.rf.fit(x_train, y_train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            rf_x = torch.tensor(
                self.rf.predict(x_np),
                dtype=torch.float32,
                device=x.device,
            )
        return rf_x
