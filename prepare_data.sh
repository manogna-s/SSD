cd data
git clone https://github.com/gulvarol/grocerydataset.git annotations
wget https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz
tar -xvzf ShelfImages.tar.gz
rm ShelfImages.tar.gz
python3 prepare_data.py
cd ..