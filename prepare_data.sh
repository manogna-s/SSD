git clone https://github.com/gulvarol/grocerydataset.git data
cd data
wget https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz
tar -xvzf ShelfImages
rm ShelfImages.tar.gz
python3 prepare_data.py
cd ..