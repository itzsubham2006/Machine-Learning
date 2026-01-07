# ML PROJECT WORKFLOWW: 

    DATA =>  DATA_PREPROCESSING =>   DATA_ ANALYSIS =>  TRAIN_TEST_SPLIT =>  MACHINE_LEARNING MODEL

    In train_test_split we take 80-90% of the data to train (training-data) the model and rest of the data is used for testing the model(testing-data).





# SUPERVISED LEARNING : The algorithm learns the pattern from the labelled data.
    TYPES : CLASSIFICATION AND REGRESSION

        CLASSIFICATION :                                                            REGRESSION :
            1. Logistic Regression                                              1. Linear Regression
            2. Support Vector Machine classifier(SVM).                          2. Laso Regression
            3. Decision Tree                                                    3. Polynomial Regression
            4. K-Nearest Neighbor                                               4. Support Vector Machine Regression
            5. Random Forest                                                    5. Random Forest Regressor
            6. Naive Bayes Classifier                                           6. Bayesian Linear Regression



# UNSUPERVISED LEARNING : The machine learning algorthim learns from the unlabelled data.
    TYPES : CLUSTERING AND ASSOCIATION. 
        Clustering : It is an unsupervised task that involves of grouping the similar data points.
        Association : It is an unsupervised task that is used to find the important relationships between data-points.

    MODELS : 
    1. K-Means Clustering.
    2. Hierarchial Clustering.
    3. Principal Component Analysis (PCA).
    4. Apriori.
    5. Eclat



# MODEL SELECTION : Model Selection in Machine learning is the process of choosing the best model for particular problem. Selecting the models may depend on many factors like dataset, task, nature etc.
    Some type of Models are :
    1. Linear Regression, 
    2. Logistic Regression, 
    3. K-meane clustering, 
    4. Neural Network etc. 



# OVERFITTING : This is a issue mainly arise when the machine learning models learn too much from the data. Where the model try to fit every data-points resulting in Overfitting.
    Causes:     
    1. Less data
    2. Increased Complexity.
    3.  More number of complexity.

   

# UNDERFITTING : This is a issue mainly arise when the machine learning models does learn much from the data. Where the model fails to fit enough data points.


# BIAS : VARIANCE TRADE-OFF IN MACHINE LEARNING :

     BIAS : Error from wrong assumptions (model too simple).

        In underfitting there is a high bias and low variance and in Overfitting there is a high variance and low bias. -->
        If Bias = low then error = low.

     VARIANCE : Variance is the amount that estimate the target function will change if different training data was used.


# TECHINIQUES TO HAVE BETTER BIAS: 

    1. Good Model selection.
    2. Regularization.
    3. Dimensionally Reduction.
    4. Ensemble Methods.


# LOSS FUNCTION : It measures how far is the estimated value is from its true value.
    It is helpful to determine which models performs better and which parameters performs better.
    TYPES: 
    1. Cross Entropy Loss.
    2. Sqaured Error Loss.
    3. KL Divergence.




# MODEL EVALUATION: 
    Supervised Learning is of two types : Classification and Regressions.

    For 'Classification' ( predicting a class or discrete values, e.g = True/False/apple ) the evualtion metric is : Accuracy score. 
        Accuracy score is the (ratio of number of correct predictions to the total number of input data ) * 100.

    For 'Regression' the evaluation metrics is Mean squared Error.
    Mean Squared Error (MSE) measures the average of the squares of the errors, that is, the average squared difference bwtween the estimated values and the actual value. 


# PARAMETERS:
    TYPE OF PARAMETERS: 
        1. Model Parameters => These are the parameters of the model that can be determined by training with training data. These can be considered as Internal Paramters. 
        (Weights and Bias).

            Weight : It decides how much influence the input data has on the output data.


        2. Hyperparameters => Hyperparamaters are the paramters whose value control the learning process. These are adjustable (we can set)  paramters used to obtain an optimal model. External Parameters. 
        Learning Rate and Number of Epoche.

            * Learning Rate : It is a tuning paramter in an optimization algorthims that determines the step size at each iteration while moving toward minimmum loss function.
            * Number of Epoche : It represent the number of iteration over the entire dataset.
    
 


# GRADIENT DESCENT : Gradient Descent is an optimization algorthim which is used for minimizing the loss funciton in varous machine learning algorithms. It is used for updating the parameters used in machine learning model.
w = w-a*dw
b = b-a*db


# LOGISTIC REGRESSION : Logistic Regression is a supervised machine learning algorithm used for classification problems — mainly binary classification (e.g., Yes/No, Spam/Not Spam, 0/1).

Y_cap = 1/(1+e^-z)   and z = w.X + b

y_cap = predicted value
X = independent variable
w = weight
b = bias




# SUPPOR VECTOR MACHINE (SVM): 

    // ADVANTAGES: 
        1. Works well with smaller data-sets.
        2. Works effecientyl when there is a clear margin of the separation.
        3. Works well with high dimensional data. 

    // DISADVANTAGES: 
        OPPOSITE OF ADVANTAGES.
        

    HPYERPLANE : A 2d/3d line/plane that seperates the data-points into two sets of classes.
    
    SUPPORT_VECTORS : A Support Vectors are the points which lie nearest to the hyperplane. If these data-points changes the position of the hyperplane changes.




# SVM KERNAL: 
    Kernal Function generally transform the training set of data so that a non-linear decision surface can be transformed to a linear equation in a higher number of dimension spaces. It returns the inner product between two points in a standard feature dimension. 











# LASO REGRESSION : 
    1. SUPERVISED LEARNING MODEL.
    2. REGRESSION MODEL.
    3. LEAST ABSOLUTE SHRINKAGE AND SELECTION OPERATOR.
    4. IMPLEMENTS REGULARIZATION (L1) TO AVOID OVERFITTNG. 

# REGULARIZATION : 
    Regularization is used to reduce the overfittng of the model by adding a penalty term (λ) to the model. Lasso Regression uses L1 regularization technique.
    The penalty term reduces the value of coefficient or eliminate coefficient, so that the model has fewer coefficients. As a result overliffting can be avoided. This process is called the Shrinkage. 




# K-FOLD CROSS-VALIDATION : 
    In K-fold Cross-Validation we split the data into "k" numbers of folds. One chunk of data is used as test data for evaluation & remaining part of the data is used for training the model. Each time, a different chunk will be used as the test data.


    Advantages :
    1. Better alternatives for train-test-split when the data-set is small.




# HYPER-PARAMETER TUNING: 
    Hyperparamter tuning refers to the process of choosing the optimum set of hyperparamters for a Machine Learning model. This process also called Hyperparamter Optimization. 
    




# ACCURACY SCORE IS NOT RELIABLE FOR UNEVEN DATA-SET, TO PREVENT THAT WE USED CONFUSION MATRIX.

    Confusion Matrix is matrix used for evaluating the performance of a Classification Model. It gives more information about the accuracy score. 

# PRECISION: 
    It is the ratio of True Positive to the sum of True Positive and False Positive. It measures, out of the total predicted positive, how many are actually poisitive.