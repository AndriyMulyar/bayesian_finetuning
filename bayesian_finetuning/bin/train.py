# pylint: disable=unused-argument, arguments-differ, arguments-differ, invalid-name
'''
Main trainer for models.
'''

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import interpretable_splicing.models as models
from interpretable_splicing.datamodules.splicing_prediction import SplicingPredictionDatamodule
from interpretable_splicing.utils.parse_config import parse_config

if __name__ == '__main__':
    """
    Processes training arguments from the command line and a config file.
    The config file specifies what type of model we are training and overrides any command line arguments.

    python bayesian_finetuning/bin/train.py --help
    """

    args = parse_config()

    model_class = getattr(models, args.model)

    pl.seed_everything(args.seed)

    raise NotImplementedError("Not implemented yet.")

    # data module
    datamodule = SplicingPredictionDatamodule(
        train=(args.train_x_path, args.train_y_path),
        val=(args.val_x_path, args.val_y_path),
        test=(args.test_x_path, args.test_y_path),
        batch_size=args.batch_size,
        num_workers=args.data_workers,
        multi_gpu=args.gpus > 1,
    )

    datamodule.setup('fit')

    model = model_class(vars(args))

    wandb_logger = WandbLogger(project=args.wandb_project_name)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    # model.build_metrics()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_acc',
    #     dirpath='model/',
    #     filename=args.model + '_{epoch:02d}_{val_acc:.3f}',
    #     save_top_k=3,
    #     mode='max',
    # )

    from pytorch_lightning.callbacks import GradientAccumulationScheduler

    batch_size_scheduler = GradientAccumulationScheduler(scheduling={2*k:2**k for k in range(1,10)})

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_monitor, batch_size_scheduler], #checkpoint_callback
        logger=wandb_logger,
        precision=args.precision,
        stochastic_weight_avg=True
    )

    trainer.fit(model, datamodule)
