import logging

from dltranz.data_load.data_module.coles_data_module import ColesDataModuleTrain
from dltranz.data_load.data_module.cpc_data_module import CpcDataModuleTrain
import pytorch_lightning as pl
from dltranz.lightning_modules.coles_module import CoLESModule
from dltranz.lightning_modules.cpc_module import CpcModule
from dltranz.util import get_conf

logger = logging.getLogger(__name__)


def main(args=None):
    conf = get_conf(args)

    if 'seed_everything' in conf:
        pl.seed_everything(conf['seed_everything'])

    data_module = None
    for m in [ColesDataModuleTrain, CpcDataModuleTrain]:
        if m.__name__ == conf['params.data_module_name']:
            data_module = m
            break
    if data_module is None:
        raise NotImplementedError(f'Unknown data module {conf.params.data_module_name}')
    logger.info(f'{data_module.__name__} used')

    pl_module = None
    for m in [CoLESModule, CpcModule]:
        if m.__name__ == conf['params.pl_module_name']:
            pl_module = m
            break
    if pl_module is None:
        raise NotImplementedError(f'Unknown pl module {conf.params.pl_module_name}')
    logger.info(f'{pl_module.__name__} used')

    model = pl_module(conf['params'])
    dm = data_module(conf['data_module'], model)
    trainer = pl.Trainer(**conf['trainer'])
    trainer.fit(model, dm)

    if 'model_path' in conf:
        trainer.save_checkpoint(conf['model_path'], weights_only=True)
        logger.info(f'Model weights saved to "{conf.model_path}"')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)
    import torch.autograd.profiler as profiler

    main()
