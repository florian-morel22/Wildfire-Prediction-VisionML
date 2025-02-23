PART=ENSTA-l40s #ENSTA-h100 #ENSTA-l40s
TIME=02:00:00
CLUSTER = srun --pty --time=$(TIME) --partition=$(PART) --gpus=1

DATA_PATH="./data"
METHOD="ss_selftraining_resnet"

N_CLUSTERS=4
CLUSTERING_ALGO="kmeans"

PARAMS = --data_path=$(DATA_PATH)\
	--n_clusters=$(N_CLUSTERS)\
	--clustering_algo=$(CLUSTERING_ALGO)
	
NUM_SAMPLES = 500 # Number of samples used in DEBUG mode.

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
	$(CLUSTER) python main.py --method=$(METHOD) $(PARAMS)

debug:
	$(CLUSTER) python main.py --DEBUG --method=$(METHOD) $(PARAMS) --num_samples=$(NUM_SAMPLES)

test:
	$(CLUSTER) python main.py --DEBUG --method=all $(PARAMS) --num_samples=5
