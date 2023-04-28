import logging
import torch
import pytorch_lightning as pl
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
# nohup python3 train.py > train.txt &

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

@hydra.main(version_base=None, config_path="configs", config_name="train_basic")
def _train(cfg: DictConfig):
    cfg.trainer.enable_progress_bar = True
    return train(cfg)

def train(cfg: DictConfig) -> None:
    # Delayed imports to get faster parsing
    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info("Preparing data")
    data_module.prepare_data()
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    logger.info("Loading model")
    model = instantiate(cfg.model)

    logger.info(f"Model '{cfg.model.modelname}' loaded")

    logger.info("Loading callbacks")
    metric_monitor = {
        # Set metrics in Dict formet to monitor
        # 'coords_loss': 'coords_loss',
        # 'types_loss': 'types_loss',
        # 'overall_loss':'overall_loss',
        # 'acc@1': 'acc@1',
        # 'acc@5': 'acc@5',
        # 'acc@10': 'acc@10'
    }

    callbacks = [
        pl.callbacks.RichProgressBar(),
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.latest_ckpt),
        instantiate(cfg.callback.last_ckpt)
    ]
    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=None,
        callbacks=callbacks
    )
    logger.info("Trainer initialized")

    logger.info("Fitting the model..")
    trainer.fit(model, datamodule=data_module)
    logger.info("Fitting done")

    trainer.test(model, datamodule=data_module)    
    logger.info("Testing done")

if __name__ == '__main__':
    _train()
