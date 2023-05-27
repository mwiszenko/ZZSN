from torch.utils.data import DataLoader

from zzsn.base import CustomImageDataset, create_data_loader, create_dataset


def main():  # pragma: no cover
    ds_train: CustomImageDataset = create_dataset("train")
    ds_val: CustomImageDataset = create_dataset("val")
    ds_test: CustomImageDataset = create_dataset("test")
    dl_train: DataLoader = create_data_loader(ds_train, "train")
    dl_val: DataLoader = create_data_loader(ds_val, "val")
    dl_test: DataLoader = create_data_loader(ds_test, "test")
