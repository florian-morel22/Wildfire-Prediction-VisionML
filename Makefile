

PART=ENSTA-h100 #ENSTA-h100 #ENSTA-l40s
TIME=01:00:00

DATA_PATH="./data"
METHOD="vit" #vit #basic_cnn #clustering_vit #all

NB_CLUSTERS=2
CLUSTERING_ALGO="kmeans"

PARAMS = --data_path=$(DATA_PATH)\
	--nb_clusters=$(NB_CLUSTERS)\
	--clustering_algo=$(CLUSTERING_ALGO)


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
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py --DEBUG --method=$(METHOD) $(PARAMS)

test:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py --DEBUG --method=all $(PARAMS)