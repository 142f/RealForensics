import logging
import os
import sys

import hydra
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STAGE2_ROOT = os.path.dirname(os.path.realpath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if STAGE2_ROOT not in sys.path:
    sys.path.insert(0, STAGE2_ROOT)
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from stage2.combined_learner import CombinedLearner
from stage2.data.combined_dm import DataModule

# static vars
logging.getLogger("lightning").propagate = False
__spec__ = None


@hydra.main(config_path="conf", config_name="config_combined")
def main(cfg):
    cfg.gpus = torch.cuda.device_count()
    if cfg.gpus < 2:
        cfg.trainer.accelerator = None

    learner = CombinedLearner(cfg)
    data_module = DataModule(cfg, root=REPO_ROOT)

    if cfg.model.weights_filename:
        if os.path.isabs(cfg.model.weights_filename) or os.path.exists(cfg.model.weights_filename):
            df_weights_path = cfg.model.weights_filename
        else:
            df_weights_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "weights", cfg.model.weights_filename
            )
        state_dict = torch.load(df_weights_path)
        weights_backbone = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
        learner.model.backbone.load_state_dict(weights_backbone)
        weights_df_head = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
        learner.model.df_head.load_state_dict(weights_df_head)
        print("Weights loaded.")

    # Train
    trainer = Trainer(**cfg.trainer)
    trainer.test(learner, datamodule=data_module)


if __name__ == "__main__":
    seed_everything(42)
    main()
