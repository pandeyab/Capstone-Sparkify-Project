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
    Logistic Regression
    Gradient Boosting
    
   I chose these above 3 methods to train on our dataset due to the nature of dataset. It falls under Supervised Learning and all these methods
   are well fitted and suitable for Supervised Learning Binary Classification. Since the datset is known and its better to apply decision tree algorithms where trees
   get added sequentially and they try to improve the performance of the model : - Random Forest and Gradient Boosted Trees.
   Logistic regression method uses linear model for prediction which is also best suited for scenarios like this.
   
   I split full dataset into training and test data and perform these algorithm one by one. Accuracy and F1- score were the two evaluation parameters. And trained our 3 
   models on train data and done the prediction.

3. Tuning :

   Later I tuned the LR model for improvement.
   
   .
### Results <a name = "results"></a>

I split full dataset into training and test data and perform these algorithm one by one. Accuracy and F1- score were the two classification/evaluation parameters I chose for this project.
	
I chose them as they are best suited for Supervised Learning Binary Classification problems. To get the performance of the model, in binary classification we have "Confusion Metrix" which consists True Positive, True Negative, False Positive and False Negative. Accuracy and F1-score are derived from this metric in order to calculate the performance of the model.

	**Accuracy = (TP + TN)/ (TP+TN+FP+FN)  →
	It is basically a measure of all the correctly classified points. If all the points are correctly classified we will get good Accuracy but it has shortcomings also. When 	dataset are improper, the prediction can't be right which will be deriving from accuracy.
	
	**F1-score = 2*P*R/(P+R) where P (Precision) = TP/(TP+FP), R(recall) = TP/(TP+FN) -->
	 It take care of both Precision and Recall which basically tells us that how the output of model is useful and how complete the output are respectively. So F1-score 		helps Accuracy or complements it incase the points are incorrectly classified.

In order to support Accuracy, I used F1-score. Because both can take care of correctly and incorrectly classified points collectively. However, main parameter for this experiment is F1-score.

Below are the results after training and prediction from models:


	F1 Score for Random Forest Classifier is 0.706
	The accuracy for Random Forest Classifier is 0.76
	Time taken : 297.50581884384155 sec
	
	F1 Score for Logistic Regression is 0.9592380952380952
	The accuracy for Logistic Regression is 0.96
	Time taken : 323.0742998123169 sec
	
	F1 Score for Gradient Boosted Trees is 0.613056133056133
	The accuracy for Gradient Boosted Trees is 0.64
	Time taken : 435.00629568099976 sec
	
	
In above results from 3 different methods we could see that the F1-score and Accuracy of Logistic Regression method is much much better compared to other 2. It however took a bit more training and prediction time than Random Forest method. It will be taking more computation time if the size of dataset is larger.
Next step is to tune and optimize Random Forest and Gradient Boosting Tree models and see how do they fare with respect to Accuracy and F1-score metrics of Logistic Regression output. I will compare the result of tuned model against LR and see which one is doing better.

As of now LR is performing way better than other 2 models.

In this section I tried to tune hyperparameters of Random Forest Classifier and Gradient Boosted Tree models. Since Logistic Regression is giving Accuracy and F1-socre close to 9%, I felt there is no need to tune this model. But It can be tuned to reduce the computational time.
1st I started with Random Forest Classifier where I used GridSearch and implemented impurity measures by using 'gini' and 'entropy'. Result is not yet so impressive. But It can be improved by experimenting different hyperparameters.

	The accuracy for tuned Random Forest is 0.56
	The F-1 Score for tuned Random Forest is 0.5516190476190476
	

later I tuned hyperparameters to Gradient Boosted tree method.
	
	The accuracy for tuned Gradient Boosted tree is 0.72
	The F-1 Score for tuned Gradient Boosted tree is 0.6743123543123543

In the end I can see Logistic Regression performed much better than other 2 algorithms. 
It produces Accuracy and F1 Score of 96% approx. Since the engineered features were less, approx 225, it is still difficult to say that this conclusion will stand true in general. It might changes with large dataset but still the winner among three selected models is Logistic Regression for this project.

I published the findings at below location:
https://medium.com/@abhis197/user-churn-prediction-for-sparkify-udacity-dsnd-capstone-project-a6ef6f21a7c6 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
 Udacity has the final lcensing on this project and Sparkify.com for dataset. 
 I would like to thanks Udacity for providing this interesting project to work on.

