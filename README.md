### query_tag_prediction

This project aims to predict a tag for a given query. 
There are 3 tags available in the dataset and for the purpose of 
this project we'll be focusing on 'tag1'.  

The entire program has been build and tested locally before training on 
google colab (due to extra compute resources required to run a Bert model)

The program has been developed with Docker (in order to isolate requirements from packages).
In order to use/run local_notebook you need to follow the below steps:

1. Install Docker 
2. change directory to 'query_tag_prediction/images' and run 'docker compose up' in terminal window - 
   This will build the docker image (if doesn't already exist) and provide a localhost
   link to run jupyterlab.
   
In order to view the overall project and outputs, please refer to colab_notebook.

Please refer to https://github.com/asingh86/query_tag_prediction.git for the most up to date code.