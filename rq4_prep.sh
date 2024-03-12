######## DOWNLOAD MODELS ########
# Table detection
if [ ! -f pubtables1m_detection_detr_r18.pth ]; then
	echo "Downloading pubtables1m_detection_detr_r18.pth"
	wget https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth
fi
# Table structure recognition
if [ ! -f TATR-v1.1-All-msft.pth ]; then
	echo "Downloading TATR-v1.1-All-msft.pth"
	wget https://huggingface.co/bsmock/TATR-v1.1-All/resolve/main/TATR-v1.1-All-msft.pth
fi

######## Table model Infererence ########
for lang in en de es fr it ja pt zh; do python -m tabletransformer.inference --lang ${lang} --dataset_mode train --mode extract -clmpoz; done
for lang in en de es fr it ja pt zh; do (python -m tabletransformer.inference --lang ${lang} --dataset_mode val --mode extract -clmpoz &); done


######## VRDSynth ########
rel_type=legacy_table;thres=1.0;hops=$2;for lang in $1;do python -m methods.decisiontree_ps_entity_linking --upper_float_thres 1.0 --rel_type=${rel_type} --hops ${hops:=3} --lang ${lang} --mode train --strategy=precision_counter;done