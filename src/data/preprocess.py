import h5py

from pathlib import Path
from src.tools.utils import collect_fpaths
from tqdm import tqdm

from PIL import Image
from torchvision import transforms as T
import numpy as np


def ims_2_hdf5(src, dst, formats=["png", "jpg", "jpeg"], channels=3, size=(512, 512)):
    im_paths = collect_fpaths(src, formats)
    num_samples = len(im_paths)

    with tqdm(total=num_samples) as pbar:
        ims = np.empty((0, *size, channels))
        for im_path in im_paths:
            im = Image.open(im_path).convert("RGB")
            im_tfm = np.asarray(T.Compose([T.CenterCrop(min(im.size)),
                                           T.Resize((512, 512))])(im))
            ims = np.append(ims, np.asarray([im_tfm]), axis=0)
            pbar.update()

    with h5py.File(dst, "w") as out:
        out.create_dataset("im", (num_samples, *size, channels), dtype='u1', data=ims)


# todo: this
# if __name__ == "__main__":
    # read args
    # split tp train and val




