# !/bin/bash

git clone https://github.com/ramimatar/MSSFinalProj.git

pip install awscli

aws configure set aws_access_key_id AKIAT4LS2WDTOZD4ZD6K
aws configure set aws_secret_access_key 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
aws configure set default.region us-east-2

cd MSSFinalProj/proj/

mkdir musdb

aws s3 cp s3://mymusicdatasets/musdb18.zip musdb/

sudo apt-get install unzip

unzip musdb18.zip

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 --force-reinstall

pip install pytorch-lightning

pip install musdb

pip install museval

pip install tensorboard

