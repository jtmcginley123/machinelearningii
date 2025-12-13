### we need to import the pandas and numpy (essential libraries)
import pandas as pd
import numpy as np
### Import data visualization packages 
import matplotlib.pyplot as plt 
import seaborn as sns 
import altair as alt 
### import preprocessing packages
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score ### needed to create training and testing datasets 
from sklearn.impute import SimpleImputer, KNNImputer ### need to impute the data 
from sklearn.preprocessing import StandardScaler, OneHotEncoder ### need to standarize and scale the data correctly 
from sklearn.compose import ColumnTransformer ### this is needed to create the pipeline 
from sklearn.pipeline import Pipeline ### this will create the pipeline 
from sklearn.tree import DecisionTreeClassifier, plot_tree , DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score, recall_score, precision_recall_curve,precision_score, f1_score, roc_auc_score, confusion_matrix


random_seed = 172193
np.random.seed(random_seed)
df = pd.read_excel('Larry/Employee_Data_Project.xlsx')
target = 'Attrition'
drop = ['EmployeeID' , 'StandardHours']
num_vars = ['Age' , 'DistanceFromHome' , 'Income' , 'NumCompaniesWorked' , 'TotalWorkingYears' , 'TrainingTimesLastYear'
            , 'YearsAtCompany' , 'YearsWithCurrManager']
ordinal_vars = ['Education' , 'JobLevel' , 'EnvironmentSatisfaction' , 'JobSatisfaction']
nom_vars = ['Gender' , 'BusinessTravel' , 'MaritalStatus']
df['Attrition_lab'] = df['Attrition'].map({'Yes':1 , 'No' :0})
X = df[num_vars+ordinal_vars+nom_vars].copy()
y= df['Attrition_lab'].copy()
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=random_seed)
## need to transform the numeric columns, nominal and ordinal columns 
### KNN imputation for the numerical columns
numeric_imputation = Pipeline(steps = [(
    'imputer' , KNNImputer(n_neighbors=5 , weights='uniform')),
('scaler' , StandardScaler())])
### regarding the nominal variables , use onehotencoding and replace the nulls with the mode (most frequent)
nominal_imputation = Pipeline(steps=[(
'imputer' , SimpleImputer(strategy= 'most_frequent')),
('one_hot_encoder' , OneHotEncoder(handle_unknown= 'ignore' , sparse_output= False , drop = 'first'))])
### regarding the ordinal variables replace with the median 
ordinal_imputation = Pipeline(steps= [('imputer' , SimpleImputer(strategy='median'))])
pre_process = ColumnTransformer(transformers=[('numeric' , numeric_imputation , num_vars),
                                              ('ordinal' , ordinal_imputation, ordinal_vars),
                                              ('nominal' , nominal_imputation , nom_vars)])
X_train_processed = pre_process.fit_transform(X_train)  
X_test_processed = pre_process.transform(X_test)
feature_names = pre_process.get_feature_names_out()
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns= feature_names)
### during the preprocessing pipeline , was converted to a np.array 
### convert to np.array to match X_train_processed

y_train_arr = y_train.values
pos_idx = np.where(y_train_arr == 1)[0]
neg_idx = np.where(y_train_arr == 0 )[0]

#### set randomseed again 
np.random.seed(random_seed)
keep_neg= np.random.choice(neg_idx, size = len(pos_idx), replace = False)
keep_idx = np.concatenate([pos_idx, keep_neg])

y_train_balanced = y_train_arr[keep_idx]
X_train_balanced = X_train_processed[keep_idx]

y_train_balanced_df = pd.DataFrame(y_train_balanced , columns= ['Attrition_lab'])
X_train_processed_balanced_df = pd.DataFrame(X_train_balanced , columns = feature_names)


### Decision Tree Classifier

tree = DecisionTreeClassifier(random_state=random_seed)
param_grid = {"ccp_alpha" : np.arange(0.001, 0.101, 0.01)}
grid = GridSearchCV(
    estimator=tree, 
    param_grid=param_grid,
    scoring= 'roc_auc',
    cv = 10, 
    n_jobs = -1)
grid.fit(X_train_processed_balanced_df, y_train_balanced)
print("Best params:", grid.best_params_)
print("Best mean CV ROC AUC:", grid.best_score_)
best_tree = grid.best_estimator_
cv_results = pd.DataFrame(grid.cv_results_)
print("\nCV results:")
print(cv_results[["param_ccp_alpha", "mean_test_score", "std_test_score"]])

### Decision Tree Regression 

pipe = Pipeline([('tree', DecisionTreeRegressor(random_state=random_seed))])
param_grid2 = {'tree__ccp_alpha' : np.arange(0.001,0.101, 0.01)}
grid2 = GridSearchCV(
    estimator = pipe,
    param_grid=param_grid2, 
    scoring = 'r2',
    cv = 10, 
    n_jobs= -1)

grid2.fit(X_train_processed_balanced_df, y_train_balanced)
print("Best Params:" , grid2.best_params_ )
print("Best CV R-Sqaured:" , grid2.best_score_)

best_pipe = grid2.best_estimator_
best_tree = best_pipe.named_steps['tree']


## Random forest
random_forest_full = RandomForestClassifier(n_estimators = 150, random_state=random_seed, n_jobs=-1)
random_forest_full.fit(X_train_processed_df , y_train)
cvs = cross_val_score(random_forest_full, X_train_processed, y_train , cv = 10 , scoring = 'f1')
print(f'cv scores mean: {cvs.mean()} , and cv scores standard deviation : {cvs.std()}')
y_probs = random_forest_full.predict_proba(X_test_processed_df)[:,1]
y_pred = (y_probs > 0.50).astype(int)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print( f' accuracy : {accuracy} , recall : {recall} , precision : {precision} , f1 : {f1}')

## get ROC-AUC 

roc_auc = roc_auc_score(y_test, y_probs)

## get confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(f'ROC-AUC : {roc_auc}')
print(cm)

### classification report: 
cr = classification_report(y_test, y_pred , target_names=['No Attrition (0)' , 'Attrition (1)'])
print(cr)

## now do it for the balanced/ downsampled dataset: 

random_forest = RandomForestClassifier(n_estimators = 150, random_state=random_seed, n_jobs=-1)
random_forest.fit(X_train_processed_balanced_df , y_train_balanced)
cvs1 = cross_val_score(random_forest, X_train_processed_balanced_df, y_train_balanced , cv = 10 , scoring = 'f1')
print(f'cv scores mean: {cvs1.mean()} , and cv scores standard deviation : {cvs1.std()}')
y_probs1 = random_forest.predict_proba(X_test_processed_df)[:,1]
y_pred1 = (y_probs1 > 0.50).astype(int)
accuracy1 = accuracy_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
f12 = f1_score(y_test, y_pred1)

print( f' accuracy : {accuracy1} , recall : {recall1} , precision : {precision1} , f1 : {f12}')

## get ROC-AUC 

roc_auc1 = roc_auc_score(y_test, y_probs1)

## get confusion matrix
cm1 = confusion_matrix(y_test, y_pred1)

print(f'ROC-AUC : {roc_auc1}')
print(cm1)

### classification report: 
cr1 = classification_report(y_test, y_pred1 , target_names=['No Attrition (0)' , 'Attrition (1)'])
print(cr1)


### Key Insight: With Random Forest: downsampling reduced precision and accuracy