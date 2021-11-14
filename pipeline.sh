echo "executing pipeline"

python -m preprocess.py -input raw_data/ -output preprocessed_data/
python -m classify.py -input preprocessed_data/ -num_epochs 10