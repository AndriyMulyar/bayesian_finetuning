import pytorch_lightning as pl
import torch
import numpy as np
import logging
import transformers
import json
from bayesian_finetuning.models.bert import GLUETransformer
from bayesian_finetuning.datamodules.glue import GLUEDataModule

from random import gauss

transformers.logging.set_verbosity_error()
pl.utilities.distributed.log.setLevel(logging.ERROR)

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def l2(model1, model2):
    return (solution - updated).pow(2).sum(0).sqrt()

if __name__ == "__main__":

    # model = GLUETransformer.load_from_checkpoint('model/finetuned_rte_mini/rte_epoch=09_accuracy=0.650.ckpt')
    #model = GLUETransformer.load_from_checkpoint('model/random_init_rte_mini/rte_epoch=06_accuracy=0.542.ckpt')
    # model = GLUETransformer.load_from_checkpoint('model/finetuned_stsb_mini/stsb_epoch=51_pearson=0.860.ckpt')
    # model = GLUETransformer.load_from_checkpoint('model/random_init_stsb_mini/stsb_epoch=58_pearson=0.176.ckpt')
    # model = GLUETransformer.load_from_checkpoint('model/finetuned_mrcp_mini/mrpc_epoch=55_accuracy=0.792.ckpt')
    model = GLUETransformer.load_from_checkpoint('model/random_init_mrcp/mrpc_epoch=11_accuracy=0.708.ckpt')

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



    output = {}
    for curve in range(1, 20):
        random_direction = torch.FloatTensor(make_rand_vector(num_params))
        points = []
        for i in np.linspace(0, 100, 200):
            updated = solution + random_direction*i

            torch.nn.utils.vector_to_parameters(updated, model.parameters())

            x = trainer.validate(model, dataloaders=datamodule.val_dataloader())
            with torch.no_grad():
                distance = l2(model, solution).item()
            points.append({**x[0], 'distance': distance})
        output[f"curve_{curve}"] = points

        with open('finetuned_mrpc.json', 'w') as fp:
            json.dump(output, fp)


