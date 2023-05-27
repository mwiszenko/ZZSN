from zzsn.base import CustomImageDataset, create_dataset


def main():  # pragma: no cover
    ds: CustomImageDataset = create_dataset("val")
    print(len(ds))
    print(ds.img_labels.head(21))
    print(ds[20][0].show())
    print("   Done")
