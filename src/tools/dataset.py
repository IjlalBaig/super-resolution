from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image
import h5py


class SRDataset(Dataset):
    def __init__(self, path, transform=None, im_mode="RGB", im_dims=(512, 512)):
        super(SRDataset, self).__init__()
        self._path = path

        self._im_mode = im_mode
        self._im_dims = im_dims
        self._n_channels = sum(1 for c in im_mode if not c.islower())

        self._transform = transform

        self._dataframes = h5py.File(path, 'r')

    def __len__(self):
        return len(self._dataframes["im"])

    def __getitem__(self, idx):
        im = Image.fromarray(self._dataframes["im"][idx])
        if self._transform is not None:
            im_tfmd = self._transform(im)
        else:
            im_tfmd = T.Compose([T.CenterCrop(min(im.size)),
                                 T.Resize(self._im_dims),
                                 T.ToTensor(),
                                 T.Normalize([0.], [1.])])(im)
        return im_tfmd