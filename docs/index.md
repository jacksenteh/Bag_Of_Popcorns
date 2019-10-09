---
layout: default
title: Movie Reviews (Kaggle)
description: ''
---

# Data exploration
The dataset consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis and was 
split into :
* labeledTrainData.tsv
* testData.tsv
* unlabeledTrainData.tsv (extra training set with no labels)

The dataset contain the following properties:
* The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. 
* For each movie, there contain multiple reviews with no more than 30 reviews in total.
* labeledTrain and testData contain 25,000 review each with labeled and unlabeled sentiments.



