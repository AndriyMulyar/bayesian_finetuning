'''Parses a config file'''

from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import ruamel.yaml


def parse_config(config: str = None):
    parser = ArgumentParser()

    if config is None:
        parser.add_argument('--config', type=str)

    parser = pl.Trainer.add_argparse_args(parser)

    args, unknown = parser.parse_known_args()

    if config is None:
        with open(args.config, "r") as ymlfile:
            train_config = ruamel.yaml.load(ymlfile, Loader=ruamel.yaml.Loader)
    else:
        with open(config, "r") as ymlfile:
            train_config = ruamel.yaml.load(ymlfile, Loader=ruamel.yaml.Loader)

    # argument precedence is config file > command line
    args = Namespace(
        **{
            **vars(args),
            **{k: v for k, v in train_config.items() if not isinstance(v, dict)},  # top level config items
            **{k: v for k, v in train_config['training'].items()},  # training configurations
            **{k: v for k, v in train_config['prediction'].items()},  # prediction configurations
            **{k: v for k, v in train_config['hparams'].items()},  # model specific hparams
        }
    )

    return args