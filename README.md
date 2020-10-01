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

The main aim of this project to predict the users churning from Sparkify app, which is a music app just like spotify. Many users listen songs  with either free account or paid acount. many users leve the service as well and Sparkify has a large dataset of log files containing all this information. From the log files from Sparkify, we need to take a smaller size of data set and manipulate it with Spark to engineer some relevant features. 
With this we will able to predict the churning of users and it could tell the users properties who are churning out off Sparkify and that will halp the streaming company to design thier app and business accordingly.

### Files Description <a name = "files"></a>

*Sparkify.ipynb* is the main notebook where we do all the preprocessing, feature engineering and modelling.

### Results <a name = "results"></a>

I seprated the json file in training and test set. For training on data set (smaller), I chose these 3 models : Random Forest, Logistic regression and Gradient Boosted Trees. I compared the performance between the three models and evaluation metrics consist two main parameter; f1-score and accuracy. 
Gradient Boosted Trees performed better if I take accuracy in consideration  han the rest models but at a same time it took longest amount of comupatation time. Due to this, I chose Random Fores Classifier as training model which F1-score and accuracy socre is pretty competitive.

*Observation from 1st attempt of training and prediction:
 
  The F-1 Score is 0.76 ;
  The accuracy is 0.706 ;


I then tuned the model and final observations are:
  The F-1 Score is 0.76
  The accuracy is 0.7326315789473684


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
 Udacity

