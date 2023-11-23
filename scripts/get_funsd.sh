# Curl to download file, override if exists
echo "A\n" | curl -L -o data_funsd.zip https://guillaumejaume.github.io/FUNSD/dataset.zip
echo "A\n" | unzip data_funsd.zip
mv dataset funsd_dataset
