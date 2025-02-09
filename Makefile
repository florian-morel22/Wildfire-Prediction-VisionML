

PART=ENSTA-l40s #ENSTA-h100
TIME=04:00:00

setup: download_dataset

	pip install -r requirements.txt

	@printf "\033[92msetup done\033[0m\n"
	

download_dataset:
	mkdir -p data
	
	curl -L -o ./data/wildfire-prediction-dataset.zip\
  	https://www.kaggle.com/api/v1/datasets/download/abdelghaniaaba/wildfire-prediction-dataset

	unzip ./data/wildfire-prediction-dataset.zip valid/*/*.jpg -d ./data

	rm ./data/wildfire-prediction-dataset.zip
	
run:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py