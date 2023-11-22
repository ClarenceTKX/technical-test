# technical-test
This repository is temporarily hosted on GH for the completion of the technical test. 

## Setting up the virtual environment 
Run the following:
```
pipenv activate
pipenv start

pip install -r ./requirements.txt
```

## Common Voice Dataset
To ensure that the dataset loads properly, please change the `local_data_path` variable (`_split_generators` class) in the loading script. (attached loading script and configuration file in main directory for reference; place loading script into parent folder of Common Voice (version attached in technical test PDF) to run)

For the asr folder, cv-train-dev's folder and 4000+ audio files were duplicated for ease of access for the other tasks as well. Please duplicate the folder and place it into `asr` if running the scripts individually.

## cv-valid-dev csv
For each task, a prefixed 'updated_' version of the `cv-valid-dev` csv file may be found in their corresponding tasks' folders. 

## PDF submissions
`training-report.pdf` and `essay-ssl` may be found in the main directory for ease of access. 

