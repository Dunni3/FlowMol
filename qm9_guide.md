run these commands to download the dataset
```console
mkdir data/qm9_raw
cd data/qm9_raw
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
unzip qm9.zip -d ./
rm qm9.zip
wget https://ndownloader.figshare.com/files/3195404 -O uncharacterized.txt
```

