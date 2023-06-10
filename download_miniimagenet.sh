mkdir -p data/miniImageNet/data

kaggle datasets download -d whitemoon/miniimagenet -p data/miniImageNet/data/
unzip data/miniImageNet/data/miniimagenet.zip -d data/miniImageNet/data/

rm -rf data/miniImageNet/data/miniimagenet.zip
