from zzsn.base import CustomImageDataset, create_dataset


def main():  # pragma: no cover
    ds_train: CustomImageDataset = create_dataset("train")
    ds_val: CustomImageDataset = create_dataset("val")
    ds_test: CustomImageDataset = create_dataset("test")
