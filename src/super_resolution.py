import src.models as models
import src.tools.utils as utils
from pytorch_lightning import Trainer
from PIL import Image
import torchvision.transforms as T
import time

from pathlib import Path


import logging
logger = logging.getLogger(__file__)


def super_resolution(opts):

    trainer = Trainer(gpus=opts.gpu_count,
                      default_save_path=opts.log_path,
                      precision=opts.precision,
                      accumulate_grad_batches=opts.accumulate_batches,
                      max_epochs=opts.epochs)

    if opts.mode == "train":
        kwargs = dict(train_path=opts.train_path,
                      val_path=opts.val_path,
                      lr=opts.learning_rate,
                      batch_size=opts.batch_size,
                      num_workers=opts.num_workers)
        model = getattr(models, opts.model)(**kwargs)
        if opts.ckpt_path is not None:
            logger.info("loading checkpoint: %s", opts.ckpt_path)
            model = model.load_from_checkpoint(checkpoint_path=opts.ckpt_path, **kwargs)

        trainer.fit(model)

    elif opts.mode == "test":
        device = "cuda" if opts.gpu_count else "cpu"
        model = getattr(models, opts.model)()
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

