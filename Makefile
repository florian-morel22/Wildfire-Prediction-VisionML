

PART=ENSTA-h100 #ENSTA-h100 #ENSTA-l40s
TIME=04:00:00

DATA_PATH="./data"
METHOD="basic_cnn" #vit #basic_cnn #clustering_vit #all #advanced_clustering #advanced_clustering1

N_CLUSTERS=5
CLUSTERING_ALGO="kmeans"

PARAMS = --data_path=$(DATA_PATH)\
	--n_clusters=$(N_CLUSTERS)\
	--clustering_algo=$(CLUSTERING_ALGO)
	
NUM_SAMPLES = 300

setup: download_dataset

	pip install -r requirements.txt

	@printf "\033[92msetup done\033[0m\n"
	

download_dataset:
	mkdir -p data
	
	curl -L -o ./data/wildfire-prediction-dataset.zip\
  	https://www.kaggle.com/api/v1/datasets/download/abdelghaniaaba/wildfire-prediction-dataset

	unzip ./data/wildfire-prediction-dataset.zip -d ./data

	rm ./data/wildfire-prediction-dataset.zip
	
run:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py --method=$(METHOD) $(PARAMS)

debug:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py --DEBUG --method=$(METHOD) $(PARAMS) --num_samples=$(NUM_SAMPLES)

test:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py --DEBUG --method=all $(PARAMS) --num_samples=5