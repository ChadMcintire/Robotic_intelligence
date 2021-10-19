pip3 install --upgrade pip
pip3 install tensorflow_datasets

ds = tfds.load('sun397', split='train', shuffle_files=True)
