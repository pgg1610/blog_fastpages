---
toc: true
# sticky_rank: 1
layout: post
description: Curated list of resources for data, science, and navigating life
categories: [exploratory-data-analysis, machine-learning, resources, chemical-science]
title: Model building checklist
---

## Things to consider when building a ML model


### Exploratory Data Analysis (EDA)

1. Learn about your data 
* Type of the data being analyzed  
* Is the data set, sensor measure-reading what you think it is measuring
* Dsitribution of the data / target 
* Outliers in the data 

2. Single variables 
* Type -- continuous, discreet, categorical 
* Distribution in the data 
* Scale of each variable 
* What is the resolution I am interested in? 
* Outliers in the variables 

3. Feature correlations
* Start simple -- linear correlation 
* Use domain knowledge and see if they make sense 
* Look at subset of the data to make it tractable / subsampling 

4. Selection and feature engineering 
* Make new (better?) features combining the orginal features 
* Recast, resample, forward difference, simple arthimatic operations 

### Get your feet wet  

* Split in train and test -- look at target prop statistics 
* Split train into CV or train / validation 
* Train models on the training data: 
        - Linear model
        - Non linear models 
        - Ensemble models 
        - Decision models 
* Model hyperparameters 

### Understand the model predictions, hyperparameters 

* Train on full training dataset 
* Avenues of data bleed 
* Split quality - is the train/validation data representative of test data / real-life data?  


### Points to think about When reviewing the ML method:

1. What is the source of the data (database, publication, direct experiment)? 

2. How many data points are in the training, validation and test sets? 

3. How were the sets split? Is any bias being introduced based on the type of split? 

4. Are the data, including the data splits used, released in a public forum? 

5. How were the data encoded and preprocessed for the ML algorithm?

6. How many parameters (p) are used in the model? 

7. How many features (f) are used as input? 

8. Is p much larger than the number of training points and/or is f large?

9. Which overfitting prevention techniques used?

10. Are the hyperparameter configurations, optimization schedule, model files and optimization parameters reported? 

11. Is the model black box or interpretable? 

12. Is the model classification or regression?

13. How much time does a single representative prediction require on a standard machine?

14. Is the source code released? 

15. How was the method evaluated?

16. Which performance metrics are reported?

17. Was a comparison to publicly available methods performed on benchmark datasets?

18. Do the performance metrics have confidence intervals?

19. Are the raw evaluation files available? 

