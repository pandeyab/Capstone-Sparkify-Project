# Capstone Sparkify project

Udacity DSND project 7 - Capstone Sparkify project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Files Description](#files)
4. [Process](#process)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


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

The main aim of this project to predict the users churning from Sparkify music streaming service, which is a music app just like spotify. Many users listen songs  with either free account or paid acount. Many users leave the service as well and Sparkify has a large dataset of log files containing all this information. From the log files, we need to take a smaller size of data set and manipulate it with Spark to engineer some relevant features. 
With this we will able to predict the churning of users and it could tell the users properties who are churning out off Sparkify and that will halp the streaming company to design thier app and business accordingly in order to prevent the churn.

### Files Description <a name = "files"></a>

*Sparkify.ipynb* - main workspace where all the project work are done such as EDA, Feature Engg, Modeling and Tuning to get the Churn Prediction Engine.

### Process <a name = "prcoess"></a>

1. EDA and Feature Engineering : 

   After doing cleaning and exploratory data analysis (exploring raw data and prepared the data)I moved to next step : feature engineering. 
   In feature engineering, my main aim is to building out the features which I found promising to train the models by extracting them from EDA. Here I created some 
   new features based on existing feature because I felt that these features will have the necessary and missing information which will be important for Machine Learning 
   model development. 
   Next, I aggregated the features by doing vectorization then standardizing input features by doing scaling operation of them all.

2. Modeling :

   Once I had the necessary aggregated features, my next goal was to develop model for churn prediction.
   For that reason I did some experiment and chose 3 algorithms for training and prediction.
    Random Forest Classification
    Logical regression
    Gradient Boosting
   I split full dataset into training and test data and perform these algorithm one by one. Accuracy and F1- score were the two evaluation parameters. And trained our 3 
   models on train data and done the prediction.

3. Tuning :

   Later I tuned the LR model for improvement.
   
   .
### Results <a name = "results"></a>

I split full dataset into training and test data and perform these algorithm one by one. Accuracy and F1- score were the two evaluation parameters.

Below are the results:
	F1 Score for Random Forest Classifier is 0.6435087719298245
	The accuracy for Random Forest Classifier is 0.68
	Time taken : 295.40548968315125 sec
	
	F1 Score for Logistic Regression is 0.6743123543123543
	The accuracy for Logistic Regression is 0.72
	Time taken : 130.5927119255066 sec
	
	F1 Score for Gradient Boosted Trees is 0.613056133056133
	The accuracy for Gradient Boosted Trees is 0.64
	Time taken : 786.54 sec
	
	
In above results from 3 different methods we could see that the f1-score and accuracy of Logistic Regression method is better compared to other 2. And it took lesser time in training and predicting. But It will be more if the size of dataset is larger.
Other 2 methods took more computation time compared to LR and also F1 score of those are also lesser than LR . So we will proceed with LR and will try to tune this model.

Later, I tried to use GridCV to tune and optimize the LR and improve the overall Churn prediction model. I used ParamGridBuilder and CrossValidator while setting Grid-Search Cross-Validation to train and tune the model.
In LR method, to avoid overfitting, I considered to tune “elasticNetParam” and “regParam”.


I published the finding at below location:
https://medium.com/@abhis197/user-churn-prediction-for-sparkify-udacity-dsnd-capstone-project-a6ef6f21a7c6

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
 Udacity has the final lcensing on this project and Sparkify.com for dataset. 
 I would like to thanks Udacity for providing this interesting project to work on.

