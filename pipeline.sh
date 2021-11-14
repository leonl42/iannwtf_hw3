echo "executing pipeline"

python -m preprocess.py -input genes/ -output preprocessed/
python -m classify.py -input preprocessed/ -num_epochs 20