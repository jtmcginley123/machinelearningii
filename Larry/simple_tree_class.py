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

### since first model was way to complex 

pipe = Pipeline([
    ('tree' , DecisionTreeClassifier(random_state=random_seed))
])

param_grid = {
    'tree__max_depth' : [4,5],
    'tree__min_samples_leaf' :  [15,20,25],
    'tree__ccp_alpha' : [0.001,0.004,0.01]
}
grid = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid,
    scoring = 'roc_auc',
    cv = 10,
    n_jobs=-1
)

grid.fit(X_train_processed_balanced_df , y_train_balanced)
print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV ROC AUC: {grid.best_score_:.4f}")

### Extract best tree
best_tree = grid.best_estimator_
best_dt = best_tree.named_steps['tree'] 

print(f"\nTree Complexity:")
print(f"  Leaves: {best_dt.get_n_leaves()}")
print(f"  Depth:  {best_dt.get_depth()}")

### Evaluate on test set
true_idx = np.where(best_dt.classes_ == 1)[0][0]

# Training metrics
train_probs = best_tree.predict_proba(X_train_processed_balanced_df)[:, true_idx]
train_auc = roc_auc_score(y_train_balanced, train_probs)

# Test metrics
test_probs = best_tree.predict_proba(X_test_processed_df)[:, true_idx]
test_pred = (test_probs > 0.5).astype(int)

test_auc = roc_auc_score(y_test, test_probs)
test_recall = recall_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)

print(f"\nPerformance:")
print(f"  Train AUC: {train_auc:.4f}")
print(f"  Test AUC:  {test_auc:.4f}")
print(f"  Gap:       {train_auc - test_auc:.4f} (smaller = less overfitting)")
print(f"  Recall:    {test_recall:.1%}")
print(f"  Precision: {test_precision:.1%}")

### ROC Curve (train vs test)
train_fpr, train_tpr, _ = roc_curve(y_train_balanced, train_probs)
test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)

plt.figure(figsize=(6, 6))
plt.plot(train_fpr, train_tpr, label=f"Train AUC = {train_auc:.3f}")
plt.plot(test_fpr, test_tpr, label=f"Test AUC = {test_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(22, 12))
plot_tree(
    best_dt,
    feature_names=X_train_processed_balanced_df.columns.tolist(),
    class_names=['Stay', 'Leave'],
    filled=True,
    rounded=True,
    fontsize=11,
    proportion=True  # Shows percentages instead of counts
)
plt.title(f'Decision Tree for Employee Attrition\n'
          f'({best_dt.get_n_leaves()} leaves, depth {best_dt.get_depth()}, Test AUC={test_auc:.3f})', 
          fontsize=16, fontweight='bold')
plt.show()