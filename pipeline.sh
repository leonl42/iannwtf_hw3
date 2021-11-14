echo "executing pipeline"

# set working directory to file directory
cd "$(dirname "$0")"
python -m preprocess -input raw_data/ -output preprocessed_data/
python -m classify -input preprocessed_data/ -num_epochs 10