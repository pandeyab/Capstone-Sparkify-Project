
# Udacity Data Scientist Nanodegree Capstone Project : Sparkify Project

## Table of Contents

1. [Dependencies](#dependencies)
2. [Project Motivation](#motivation)
3. [Files Description](#description)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)


### Dependencies <a name = "dependencies"></a>

All the following libraries are needed to implement this project:

**Python**<br>
**Pandas**<br>
**Matplotlib**<br>
**Seaborn**<br>
**PySpark**<br>
**Spark**<br>

It is also highly recommended to use the Anaconda distribution of Python which has most of the data science libraries preinstalled, to utilize Spark you can use the Databricks environment or AWS.

### Project Motivation <a name = "motivation"></a>

This project is a part of the Udacity's Data Scientist Nanodegree program analyzing the behaviour of users for an app called Sparkify to predict user churn. Sparkify is an app similar to Spotify and the dataset contains user behaviour log for the past few months. It contains some basic information about the users as well as information about a particular action they have taken.
A user can have multiple actions which leads to multiple entries for a user, we can identify when a user churned through the action of account cancellation.

### Files Description <a name = "description"></a>

*Sparkify.ipynb* is the main notebook where we do all the preprocessing, feature engineering and modelling.

### Results <a name = "results"></a>

We trained three different models on the dataset which are Random Forest, Support Vector Machines and Gradient Boosted Trees respectively. We compared the performance between the three models and Gradient Boosted Trees outperformed the rest by a large margin.
The metric we used to evaluate performance is F-1 Score as that gives us a better representation of the model performance. 

The final metrics for our Gradient Boosted Trees Classifier are as follows: <br>
The F-1 Score is 0.8695652173913043 <br>
The accuracy is 0.8665247795682578 <br>

Check out my blog post by clicking this [link](https://medium.com/@areddy3/predict-customer-churn-using-pyspark-c862881617b2) for more detailed analysis.

### Acknowledgements <a name = "acknowledgements"></a>

Udacity has to be acknowledged for giving me this wonderful project.

