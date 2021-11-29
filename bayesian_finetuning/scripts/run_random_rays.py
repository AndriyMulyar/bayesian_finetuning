import pytorch_lightning as pl
import torch
import numpy as np
from bayesian_finetuning.models.bert import GLUETransformer
from bayesian_finetuning.datamodules.glue import GLUEDataModule

from random import gauss

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def l2(model1, model2):
    return (solution - updated).pow(2).sum(0).sqrt()
if __name__ == "__main__":

    model = GLUETransformer.load_from_checkpoint('model/rte_epoch=09_accuracy=0.650.ckpt')

    # data module
    datamodule = GLUEDataModule(
        model_name_or_path=model.hparams.model_name_or_path,
        task_name=model.hparams.task_name,
        train_batch_size=model.hparams.train_batch_size,
        eval_batch_size=model.hparams.eval_batch_size
    )
    datamodule.setup('fit')

    trainer = pl.Trainer(gpus=1,
                         enable_progress_bar=False,
                         enable_model_summary=False,
                         weights_summary=False)

    solution = torch.nn.utils.parameters_to_vector(model.parameters())
    num_params = solution.size()[0]


    random_direction = torch.FloatTensor(make_rand_vector(num_params))

    for i in np.linspace(0,100, 100):
        updated = solution + random_direction*i

        torch.nn.utils.vector_to_parameters(updated, model.parameters())

        x = trainer.validate(model, dataloaders=datamodule.val_dataloader())
