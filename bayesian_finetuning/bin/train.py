# pylint: disable=unused-argument, arguments-differ, arguments-differ, invalid-name
'''
Main trainer for models.
'''

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from bayesian_finetuning.models import GLUETransformer
from bayesian_finetuning.datamodules.glue import GLUEDataModule
from bayesian_finetuning.utils.parse_config import parse_config

if __name__ == '__main__':
    """
    Processes training arguments from the command line and a config file.
    The config file specifies what type of model we are training and overrides any command line arguments.

    python bayesian_finetuning/bin/train.py --help
    """

    args = parse_config()

    pl.seed_everything(args.seed)

    # data module
    datamodule = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    datamodule.setup('fit')
    args.num_labels = datamodule.num_labels
    args.eval_splits = datamodule.eval_splits

    model = GLUETransformer(**vars(args))

    wandb_logger = WandbLogger(project=args.wandb_project_name)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='accuracy',
        dirpath='model/',
        filename=args.task_name + '_{epoch:02d}_{accuracy:.3f}',
        save_top_k=3,
        mode='max',
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        precision=args.precision,
    )
    trainer.fit(model, datamodule)
