rm -r transformers
git clone -b add_layoutlm_relation_extraction https://github.com/nielsrogge/transformers.git
cd transformers
python -m pip install -q ./transformers
cd ..
rm -r unilm
git clone -b layoutlmft_patch https://github.com/nielsrogge/unilm.git
