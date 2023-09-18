import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



class FirstClassModel:
    # Every method (except for performance metrics) should be defined 
    # a one line call to the appropriate method from the library you use
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear',  random_state=0)


    def fit(self,x,y):
        ## Your fit model may simply call that of sklearn or any other
        # library you wish to use (i.e. should be one line)
        self.model.fit(x, y)

    def predict_class(self,x):
        ### This method must return an array with size y
        ## whose entries are either 0 or 1#
        return self.model.predict(x)

         
    def predict_probabilities(self,x):
        ### This method must return an array with size y
        ## whose entries denote the "likelihood" that input
        # x belongs to (each) class, with shape (#data points, #classes)
        return self.model.predict_proba(x)

    def performance_metrics(self,x,y,y_pred, y_probs):  
        ### report accuracy, true positive rate, false positive rate
        ## as well as score mean and standard deviation (corresponding to class ONE)
        # ... 
        accuracy = self.model.score(x,y)
        m = confusion_matrix(y, y_pred)
        tpr=m[1,1]
        fpr=m[0,1]
        probs = y_probs[:,1]
        yprob_mean = np.mean(probs)
        yprob_stdev = np.std(probs)
        return (accuracy, tpr, fpr, round(yprob_mean,5), round(yprob_stdev,5))

# initialize model
fcm = FirstClassModel()

# read and save data
data =  pd.read_csv('./data/pa0_train.csv',index_col = False)
x,y =  np.array(data.iloc[:,:-1]), np.array(data.iloc[:,-1])

# fit data
fcm.fit(x,y)
# print(fcm.model.coef_)

# predicitions
y_probs = fcm.predict_probabilities(x)
y_pred = fcm.predict_class(x)

# performance metrics
metrics = fcm.performance_metrics(x,y,y_pred,y_probs)
print(metrics)

