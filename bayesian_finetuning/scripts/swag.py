import copy
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Callable, List, Optional, Union
from pytorch_lightning.callbacks import Callback
from bayesian_finetuning.models.bert import GLUETransformer
from bayesian_finetuning.datamodules.glue import GLUEDataModule
from bayesian_finetuning.utils.parse_config import parse_config


class GLUETransformer_SGD(GLUETransformer):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            task_name: str = None,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            random_init=False,
            **kwargs,
    ):
        super().__init__(model_name_or_path,
                         num_labels,
                         task_name,
                         learning_rate,
                         adam_epsilon,
                         warmup_steps,
                         weight_decay,
                         train_batch_size,
                         eval_batch_size,
                         eval_splits,
                         random_init,
                         **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]


class StochasticWeightAveragingGaussian(Callback):
    def __init__(self,
                 swa_epoch_start: Union[int, float] = 0.8,
                 c: int = 1,
                 k: int = 10,
                 # device: Optional[Union[torch.device, str]] = torch.device("cpu"),
                 ):
        """Implements the Stochastic Weight Averaging Gaussian (SWAG) Callback to average a model.

        Parameters
        ----------
        swa_epoch_start : TODO If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch
        annealing_epochs :
        annealing_strategy :
        avg_fn :
        device :
        """

        self._swa_epoch_start = swa_epoch_start
        # self._device = device
        self._c = c
        self._k = k

        self.theta_i_list = []  # Store \theta_i
        self.theta_bar_i_list = []
        self.theta_bar_squared_i_list = []
        self.theta_bar_squared_i_list = []  # Store \bar{\theta^2} element wise squared \theta
        self.d_hat = None
        self.sigma_low_rank = None
        self.sigma_diag = None

    def on_pretrain_routine_start(self, trainer, pl_module):
        # Store parameter \theta_0 and the one with element-wise squared
        theta_0 = torch.nn.utils.parameters_to_vector(pl_module.parameters())

        self.theta_i_list.append(theta_0)
        self.theta_bar_i_list.append(theta_0)

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self._c == 0:
            n = pl_module.current_epoch / self._c

            theta_bar_prev = self.theta_bar_i_list[-1]
            theta_i = torch.nn.utils.parameters_to_vector(pl_module.parameters())
            theta_bar_i = torch.zeros(theta_i.shape)
            theta_bar_squared_i = torch.zeros(theta_i.shape)

            for i, (theta_bar_param, param) in enumerate(zip(theta_bar_prev, theta_i)):
                theta_bar_i[i] = (n * theta_bar_param + param) / (n + 1)
                theta_bar_squared_i[i] = (n * theta_bar_param + param ** 2) / (n + 1)

            self.theta_i_list.append(theta_i)
            self.theta_bar_i_list.append(theta_bar_i)
            self.theta_bar_squared_i_list.append(theta_bar_squared_i)

    def on_train_end(self, trainer, pl_module):
        d = torch.nn.utils.parameters_to_vector(pl_module.parameters()).shape[0]
        self.d_hat = np.zeros((d, self._k))
        print(self.d_hat.shape)
        for i in range(self._k):
            try:
                self.d_hat[:, i] = (self.theta_i_list[-i - 1] - self.theta_bar_i_list[-i - 1]).detach().numpy()
            except IndexError:
                print(f"error happened at i={i}")

        self.sigma_low_rank = self.d_hat @ self.d_hat.T / (self._k - 1)
        self.sigma_diag = (self.theta_bar_squared_i_list[-1] - self.theta_bar_i_list[-1] ** 2).detach().numpy()


if __name__ == "__main__":
    model = GLUETransformer_SGD.load_from_checkpoint('rte_epoch=09_accuracy=0.606.ckpt')

    # data module
    datamodule = GLUEDataModule(
        model_name_or_path=model.hparams.model_name_or_path,
        task_name=model.hparams.task_name,
        train_batch_size=model.hparams.train_batch_size,
        eval_batch_size=model.hparams.eval_batch_size,
    )

    swag_trainer = pl.Trainer(callbacks=[StochasticWeightAveragingGaussian()], max_epochs=1)
    swag_trainer.fit(model, datamodule)

