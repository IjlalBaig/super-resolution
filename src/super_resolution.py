import src.models as models
import src.tools.utils as utils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import torchvision.transforms as T
import time

from pathlib import Path
from src.tools.utils import read_yaml
from argparse import Namespace


import logging
logger = logging.getLogger(__file__)


def super_resolution(opts):
    logger = TensorBoardLogger(save_dir=opts.log_path, name=opts.model)
    ckpt_path = Path(logger.experiment.log_dir).joinpath("checkpoint",
                                                         "{epoch}-{val_loss:.2f}").as_posix()
    checkpoint_callback = ModelCheckpoint(filepath=ckpt_path,
                                          monitor="val_loss",
                                          save_top_k=3)
    trainer = Trainer(gpus=opts.gpu_count,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      precision=opts.precision,
                      accumulate_grad_batches=opts.accumulate_batches,
                      max_epochs=opts.epochs)

    cfg_path = Path("./config").joinpath(opts.model + ".yml").expanduser().as_posix()
    model_cfg = Namespace(**read_yaml(fpath=cfg_path))
    if opts.mode == "train":

        model = getattr(models, opts.model)(model_cfg,
                                            train_path=opts.train_path,
                                            val_path=opts.val_path,
                                            num_workers=opts.num_workers)
        if opts.ckpt_path is not None:
            logger.info("loading checkpoint: %s", opts.ckpt_path)
            model = model.load_from_checkpoint(checkpoint_path=opts.ckpt_path, **opts)

        trainer.fit(model)

    elif opts.mode == "test":
        device = "cuda" if opts.gpu_count else "cpu"
        model = getattr(models, opts.model)(model_cfg)
        if opts.ckpt_path is not None:
            logger.info("loading checkpoint: %s", opts.ckpt_path)
            model = model.load_from_checkpoint(checkpoint_path=opts.ckpt_path)
        model.to(device)

        test_path = opts.test_path
        if Path(test_path).is_file():
            src_paths = [test_path]
        else:
            src_paths = utils.collect_fpaths(test_path,  ["jpg", "jpeg", "png"])

        log_path = opts.log_path
        dst_dir = Path(log_path).joinpath(time.strftime("%Y_%b_%d_%H_%M_%S"))
        dst_dir.mkdir()

        for src_path in src_paths:
            x = T.ToTensor()(Image.open(src_path)).unsqueeze(0).to(device=device)
            out = model(x)
            dst_path = dst_dir.joinpath(Path(src_path).name)
            T.ToPILImage()(out.squeeze(0).to(device="cpu")).save(dst_path)

