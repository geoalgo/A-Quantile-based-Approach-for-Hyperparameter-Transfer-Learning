import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

from sklearn.preprocessing import StandardScaler

from constants import num_gradient_updates

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from prior import Prior


def train(
        module,
        X_train: np.array,
        y_train: np.array,
        num_gradient_updates: int = num_gradient_updates,
        lr: float = 1e-2,
        num_decays: int = 3,
        factor_decay: float = 5.0,
        batch_size: int = 64,
        clip_gradient: Optional[float] = None,
        optimizer=None,
        early_stopping: bool = True,
):
    dataset = TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train)
    )
    # keep 10% of train dataset as validation
    num_train = len(dataset) * 9 // 10
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, len(dataset) - num_train])

    # dont use gpu for now
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # module = module.to(device)

    def infinite_stream():
        while True:
            # reshuffle
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for data in dataloader:
                yield data

    train_losses = []
    val_rmses = []
    first = True
    if optimizer is None:
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)

    checkpoint_freq = 100
    it = 0
    best_val_rmse = float("inf")
    checkpoint_path = Path(tempfile.gettempdir()) / f"best-model-{uuid.uuid4().hex}.pth"
    with torch.autograd.set_detect_anomaly(True):
        for _ in range(num_decays):
            with tqdm(infinite_stream(), total=num_gradient_updates, miniters=200, mininterval=2) as tqdm_iter:
                for X_batch, y_batch in tqdm_iter:
                    optimizer.zero_grad()
                    # both of shape (batch_size, output_dim,) we could also fit a covariate matrix to account
                    # the dependency between different criterion
                    mu, sigma = module(X_batch)
                    distr = torch.distributions.normal.Normal(loc=mu, scale=sigma)
                    # y_batch has shape (batch_size, output_dim)
                    loss = - distr.log_prob(y_batch).mean()
                    loss.backward()
                    loss_value = loss.item()

                    if clip_gradient is not None:
                        nn.utils.clip_grad_norm_(
                            module.parameters(),
                            max_norm=clip_gradient
                        )

                    if first:
                        def count_parameters(model):
                            return sum(p.numel() for p in model.parameters() if p.requires_grad)

                        print(
                            "\n".join(f"{name}: shape {p.shape}, {p.numel()} parameters" for name, p in
                                      module.named_parameters() if p.requires_grad)
                        )
                        print(f"number of parameters: {count_parameters(module)}")
                        first = False
                    # print(loss_value)
                    train_losses.append(loss_value)
                    optimizer.step()

                    metrics_dict = {
                        "train_loss": loss_value,
                    }

                    if it % checkpoint_freq == 0:
                        for X_val, y_val in DataLoader(val_dataset, batch_size=len(val_dataset)):
                            # compute mean
                            mu, sigma = module(X_val)
                            val_rmse = ((mu - y_val) ** 2).mean().sqrt().item()
                        metrics_dict['val_rmse'] = val_rmse
                        val_rmses.append(val_rmse)

                        if early_stopping and val_rmse < best_val_rmse:
                            # print(f" found better loss {val_rmse} than {best_val_rmse}, checkpointing in {checkpoint_path}")
                            best_val_rmse = min(best_val_rmse, val_rmse)
                            torch.save(module.state_dict(), checkpoint_path)
                        tqdm_iter.set_postfix(metrics_dict)

                    it += 1
                    if it % num_gradient_updates == 0:
                        break

        lr /= factor_decay

    if early_stopping:
        print(f"loading best model found at {checkpoint_path} with val_rmse={val_rmse}")
        module.load_state_dict(torch.load(checkpoint_path))

    return module, (train_losses, val_rmses)


class GaussianRegression(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 3, num_hidden: int = 40, dropout: float = 0.0):
        super(GaussianRegression, self).__init__()
        layers = [nn.Linear(input_dim, num_hidden)]
        for i in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.mu_proj = nn.Linear(num_hidden, 1)
        self.sigma_proj = nn.Sequential(nn.Linear(num_hidden, 1), nn.Softplus())

        def init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        # use the modules apply function to recursively apply the initialization
        self.layers.apply(init)

    def forward(self, x):
        x_hidden = self.layers(x)
        mu = self.mu_proj(x_hidden)
        sigma = self.sigma_proj(x_hidden)
        return mu, sigma


class ParametricPrior(Prior):
    def __init__(
            self,
            X_train: np.array,
            y_train: np.array,
            num_gradient_updates: int = num_gradient_updates,
            dropout: float = 0.1,
            num_layers: int = 3,
            num_hidden: int = 50,
            **train_kwargs
    ):
        super(ParametricPrior, self).__init__(
            X_train=X_train,
            y_train=y_train,
        )
        n, dim = X_train.shape
        self.scaler = StandardScaler()
        module = GaussianRegression(input_dim=dim, num_layers=num_layers, num_hidden=num_hidden, dropout=dropout)
        self.module, _ = train(
            module=module,
            X_train=self.scaler.fit_transform(X_train),
            y_train=y_train,
            num_gradient_updates=num_gradient_updates,
            **train_kwargs
        )

    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        X_test = torch.Tensor(self.scaler.transform(X))
        self.module.eval()
        mu, sigma = self.module(X_test)
        return mu.detach().numpy(), sigma.detach().numpy()
