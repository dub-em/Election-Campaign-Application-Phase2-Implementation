# Election-Campaign-Application

## Phase 2 Imiplementation(Citizens' Voice)
+ This repository is an implementation of the second Phase of the Election Campaign Application (Citizens' Voice section).  

### Purpose of this Project 
+ To automate the extraction of the dataset from the AWS postgres remote instance which was set up in the first phase of this project, transform this dataset using a local (generic) gensim corpus, predict the sentiment of each row in the extracted dataset (using this transformed data and a custom trained Simple RNN model) and adding this prediction back into the original dataset to form a new dataset.
+ To automate the extraction of trending topics from the original dataset and convert these topic and their contributing words into a tabular format.
+ To load the new dataset containing the predicted sentiment, and the tabulated trending topics back into the AWS postgres remote instance.

### Reference GitHub Repositories
+ https://github.com/dub-em/Election-Campaign-Application
+ https://github.com/dub-em/Election-Campaign-Application-Phase2
+ https://github.com/dub-em/Sentiment-Prediction-Worflow

### Duckerhub Repository
+ https://hub.docker.com/r/dub3m/citizens-voice

### Article
+ https://www.linkedin.com/pulse/citizens-sentiment-michael-igbomezie

### Contribution
This Project is open to contribution and collaboration. Feel free to connect to join the project collaborators.

### Author(s)
+ Michael Dubem Igbomezie
