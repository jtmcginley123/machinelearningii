###### will need to update variable names to make sure that the KNN pipeline dataset is used ( full not downsampled)


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
from sklearn.metrics import classification_report , accuracy_score, recall_score, precision_recall_curve,precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
################################################################################
################################################################################
#                       Start of Preprocessing                                 #
################################################################################
################################################################################
## set a random seed 
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

################################################################################
################################################################################
#                             End of preprocessing                             #
################################################################################
################################################################################



### Decision Tree Classifier

pipe = DecisionTreeClassifier(random_state=random_seed)
param_grid = {"ccp_alpha" : np.arange(0.001, 0.101, 0.01)}
grid = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid,
    scoring= 'roc_auc',
    cv = 10, 
    n_jobs = -1)
grid.fit(X_train_processed_df, y_train)
print("Best params:", grid.best_params_)
print("Best mean CV ROC AUC:", grid.best_score_)
best_tree = grid.best_estimator_
cv_results = pd.DataFrame(grid.cv_results_)

plt.figure(figsize=(14,8))
plot_tree(
    best_tree,
    feature_names = X_train_processed_df.columns.tolist(),
    class_names = ['No Attrition (0)' , 'Attrition (1)'],
    filled = True , 
    rounded = True
)
plt.show()

### get in-sample ROC curve and AUC 

probs = best_tree.predict_proba(X_train_processed_df)
true_idx = np.where(best_tree.classes_ ==1 )[0][0]
probs_true = probs[:, true_idx]
train_fpr , train_tpr , train_tresholds = roc_curve( y_train , probs_true)
train_auc_value = roc_auc_score(y_train , probs_true)
print("\nTrain ROC AUC for Attrition:", train_auc_value)
plt.figure(figsize=(6, 6))
plt.plot(train_fpr, train_tpr, label=f"AUC = {train_auc_value:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Attrition, training set")
plt.legend(loc="lower right")
plt.show()

## Variable Importance: 
importances = pd.Series(
    best_tree.feature_importances_,
    index = X_train_processed_df.columns).sort_values(ascending=False)
print(f'\nVariable Importance: {importances}')
plt.figure(figsize=(8,5))
importances.head(15).plot(kind = 'barh')
plt.gca().invert_yaxis()
plt.title("Top variable importances (Decision Tree)")
plt.xlabel("Importance")
plt.show()

### Look at two most important variables together ----

top_two = importances.head(2).index.tolist()
print("\nTop two variables:", top_two)

## create a dataset for plotting 

if len(top_two) == 2:
    var_x, var_y = top_two
    data_two_plot = X_train_processed_df.copy()
    data_two_plot['Attrition_lab'] = y_train
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data = data_two_plot,
        x=var_x,
        y=var_y,
        hue="Attrition_lab",
        alpha=0.5
    )
    plt.title(f"{var_y} vs {var_x} by Attrition")
    plt.legend(title = 'Attrition' , labels = ['No (0)' , 'Yes (1)'])
    plt.tight_layout()
    plt.show()

## predictions and the final model evaulation on the test set 

probs_test = best_tree.predict_proba(X_test_processed_df)
probs_true_test = probs_test[:, true_idx]
## get the ROC and the AUC for the test set 

test_fpr , test_tpr , test_thresholds = roc_curve(y_test, probs_true_test) 
test_auc_value = roc_auc_score(y_test, probs_true_test)

print(f'\nTst ROC AUC for Attrition: {test_auc_value}')

#### now we need to plot the ROC AUC curve

plt.figure(figsize=(8,8))
plt.plot(test_fpr , test_tpr , label = 'AUC')
plt.plot([0,1],[0,1], 'k--' , label=f"AUC = {test_auc_value:.3f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Attrition: Test Set')
plt.legend(loc='lower right')
plt.show()

### Generate preditions at a 0.5 threshold 
y_pred_test = (probs_true_test > 0.5).astype(int)
accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
print(f'Test Set -- Accuracy: {accuracy_test} , Recall: {recall_test} , Precision: {precision_test} , f1: {f1_test}')

cm_test = confusion_matrix(y_test, y_pred_test)
print(f'\nConfusion Matrix')
print(cm_test)

cr_test = classification_report(y_test, y_pred_test , target_names = ['No Attrition (0)'  , 'Attrition (1)'])
print(f'\nClassification Report')
print(cr_test)



'''

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


#### Plot the tree 

plt.figure(figsize=(14,8))
plot_tree(
    best_tree1 = best_pipe.named_steps['tree'],
    feature_names= X_train_processed_balanced_df.columns, 
    class_names = best_tree.classes_.astype("str"),
    filled = True, 
    rounded = True
)
plt.title("Decision Tree")
plt.show()
### Decision Tree Regression 


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
'''