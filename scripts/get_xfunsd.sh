# 
lang=$1
mkdir -p xfund_dataset/$lang
echo "A\n" | curl -L -o xfund_dataset/${lang}/${lang}.train.json https://github.com/doc-analysis/XFUND/releases/download/v1.0/${lang}.train.json
echo "A\n" | curl -L -o xfund_dataset/${lang}/${lang}.val.json https://github.com/doc-analysis/XFUND/releases/download/v1.0/${lang}.val.json
echo "A\n" | curl -L -o xfund_dataset/${lang}.train.zip https://github.com/doc-analysis/XFUND/releases/download/v1.0/${lang}.train.zip
echo "A\n" | curl -L -o xfund_dataset/${lang}.val.zip https://github.com/doc-analysis/XFUND/releases/download/v1.0/${lang}.val.zip
echo "A\n" | unzip xfund_dataset/${lang}.train.zip -d xfund_dataset/${lang}
echo "A\n" | unzip xfund_dataset/${lang}.val.zip -d xfund_dataset/${lang}
rm xfund_dataset/${lang}.train.zip
rm xfund_dataset/${lang}.val.zip
