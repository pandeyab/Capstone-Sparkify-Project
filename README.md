# Capstone Sparkify project

Udacity DSND project 7 - Capstone Sparkify project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Files Description](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [spark](https://spark.apache.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 


## Project Motivation <a name="motivation"></a>

This project is a part of the Udacity's Data Scientist Nanodegree program analyzing the behaviour of users for an app called Sparkify to predict user churn. Sparkify is an app similar to Spotify and the dataset contains user behaviour log for the past few months. It contains some basic information about the users as well as information about a particular action they have taken.
A user can have multiple actions which leads to multiple entries for a user, we can identify when a user churned through the action of account cancellation.

### Files Description <a name = "files"></a>

*Sparkify.ipynb* is the main notebook where we do all the preprocessing, feature engineering and modelling.

### Results <a name = "results"></a>

We trained three different models on the dataset (smaller) which are Random Forest, Support Vector Machines and Gradient Boosted Trees respectively. We compared the performance between the three models and evaluation metrics consist two main parameter; f1-score and accuracy. 
Gradient Boosted Trees outperformed the rest by a large margin but at a some time it took largest amount of comupatation time. Due to this, I chose Random Fores Classifier as training model. 

*Observation from 1st attempt of trainign and prediction:
 
  The F-1 Score is 0.76
  The accuracy is 0.706

I then tuned the model and final observations are:



## Licensing, Authors, Acknowledgements<a name="licensing"></a>
 Udacity

