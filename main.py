#!/usr/bin/env python
# coding: utf-8

# In[196]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


# In[151]:


data1 = pd.read_csv('good_doctor_data.csv')


# # Data preprocessing

# In order to idenfity key features for the diease incidence and provide recomendation for the doctor, we need to first work on our dataset and understand it better. In further parts of the case study we'll provide advanced ML techniques which could help the doctor make correct decisions. 

# In[152]:


data1.info()


# ### Comments
# 
# - The data has 3840 raws and 13 columns which as said is not a big group.
# - It could be beneficial to have the data on gender, race or socio-economical status (could be indicated like high/low/medium), the patients' medial record on other diseases like high blood pressure or diabetes, as they could bring valuable insights into the table

# In[153]:


# Overview on the data
data1.head(15)


# In[154]:


# Desc statistics
print(data1.describe())


# In[155]:


# Check for missing values in the entire DataFrame and display the rows for which the nulls are occuring
nulls_df = data1.isnull()
print("---- NUMER OF NaN VALUES WITHIN EACH COLUMN ----")
print(data1.isnull().sum())
nulls_in_record = nulls_df.sum(axis=1)
records_w_nulls = data1[nulls_in_record > 0]

#print(records_w_nulls)
print("\n---- NUMER OF 0,0 VALUES WITHIN EACH COLUMN ----")
# Number of zeros in each column
zero_counts_per_column = (data1 == 0).sum(axis=0)
print(zero_counts_per_column)


# * We're having about not more than 30 null values for blood chemistry and air quality indexes in the dataset which is quite common. They may have occured due to missing data, measurement errors, or other reasons during data collection. It's okay to drop them since they are quite scarce.
# * Apart from NaNs ther're also a lot of "0,0" values in the dataframe. It's essential to understand whether the NaNs and zeros have different meanings in the dataset. NaNs typically represent missing or unknown values, while zeros represent a specific value of zero. If they have distinct meanings, you should treat them differently during data analysis and modeling. We'd need to understand the data source to interpret the meaning of NaNs and zeros better. Medical data, in particular, may have specific conventions for representing missing values or zeros and very often it requires examination with the person responsible of data inputation. For different variables the zeros should be treated differently for sure:
#     - Preg -> Naturally, a person can have 0 pregnancies. Since we do not have the information on the patients' gender we cannot assume that those observations are a representation of males, HOWEVER we can add a separate variable on whether the patient has been pregnant at all or not as it could bring interesing results.
#     - BC1,2,3 -> Treat zeros as Zeros
#     - Blood Pressure -> could be both a mistake or a serious value depending on the hospital department where the data has been collected
#     - Skin Thickness - We can treat it as a minimal value as the values vary from 0 to 99.
#     - BMI -> For BMI to be zero, both weight and height would need to be zero, which is not a physically possible scenario for a living human being so we can treat those as NaNs
#     - GPF -> No zeros, no problem
#     - Age -> No zeros, no problem

# In[156]:


# We could also rename columns for better understanding
data1 = data1.rename(columns={'$tate': 'State', 'Air Qual\'ty Index': 'AQI','Genetic Predisposition Factor':'GPF', '# Pregnancies': 'Preg_count', 'Blood Chemestry~I': 'BC1', 'Blood Chemisty~II': 'BC2', 'Blood Chemisty~III': 'BC3'})

# Actions with zeros on the dataframe
# Create a new column 'Pregnant' to indicate whether the patient has been pregnant or not
data1['Pregnant'] = (data1['Preg_count'] > 0).astype(int)

# Treat zeros in BC1, BC2, BC3 as NaNs
bc_columns = ['BC1', 'BC2', 'BC3']

# Treat zeros in Blood Pressure as NaNs
bp_columns = ['Blood Pressure']
data1[bp_columns] = data1[bp_columns].replace(0, np.nan)

# Treat zeros in Skin Thickness as minimal value (e.g., 1)
data1['Skin Thickness'] = data1['Skin Thickness'].replace(0, 1)

# Treat zeros in BMI as NaNs
data1['BMI'] = data1['BMI'].replace(0, np.nan)


# ### Duplicates handling

# In[157]:


# Get the distinct values of 'Unique_ID' column
unique_ids = data1['Unique_ID'].nunique()

# Display the distinct values
print("Number of distinct values of 'Unique_ID' column:")
print(unique_ids)


# We can see that we have only 768 unique patients' ID which means we need to work on grouping the data under the same IDs

# In[158]:


# Group the DataFrame by 'Unique_ID' and check the number of unique values per column
unique_id_grouped = data1.groupby('Unique_ID').nunique()

# Check if the number of unique values for each column is equal to 1 for all 'Unique_ID'
all_variable_values_same = (unique_id_grouped == 1).all()

print("Are the variable values the same for each 'Unique_ID'")
print(all_variable_values_same)

# Find the 'Unique_IDs' where the variable values are not the same
ids_with_variable_variations = unique_id_grouped[unique_id_grouped != 1].index

print("\nUnique_IDs with Variable Value Variations:")
print(ids_with_variable_variations)


# In[159]:


data1.loc[data1['Unique_ID'] == 5642119]


# Sometimes for the same ID the values of some variables are different, so not to drop them we take the average of the values which are changing

# In[194]:


def transform_data(df, columns_to_keep, columns_to_average, columns_to_max):
    
    # Take the first of all values which are different under the same Unique ID
    df = df.groupby('Unique_ID')[columns_to_keep].first()

    # Calculating the average of columns under the same Unique ID
    averaged_values = df.groupby('Unique_ID')[columns_to_average].mean().reset_index()
    averaged_values = averaged_values.rename(columns=lambda col: f"Avg_{col}")
    averaged_values = averaged_values.rename(columns={'Avg_Unique_ID': 'Unique_ID'})

    # Finding the maximum of columns under the same Unique ID
    max_values = df.groupby('Unique_ID')[columns_to_max].max().reset_index()
    max_values = max_values.rename(columns=lambda col: f"Max_{col}")
    max_values = max_values.rename(columns={'Max_Unique_ID': 'Unique_ID'})

    # Merge the data
    df = pd.merge(df, averaged_values, on='Unique_ID', how='left')
    df = pd.merge(df, max_values, on='Unique_ID', how='left')

    return df

data1_transformed = transform_data(data1, columns_to_keep, columns_to_average, columns_to_max)


# In[161]:


data1_transformed.loc[data1_transformed['Unique_ID'] == 5642119]


# We can see that the output has been concolidated under the Unique ID with the added new variables on the Patient's maximum and average values of each column which have been different. Now we can proceed to dropping null values

# In[162]:


# Dropping NaNs
data1_drpd=data1_transformed.dropna()

unique_ids_dropped = data1_drpd['Unique_ID'].nunique()

# Display the distinct values
print("Number of distinct values of 'Unique_ID' column after NaNs drop:")
print(unique_ids_dropped)


# In[163]:


# Reorder the columns in the DataFrame
desired_column_order = [col for col in data1_drpd.columns if col not in ['State']] + ['State']
data1 = data1_drpd[desired_column_order]

# Create numerical_df by excluding the 'State' column
numerical_df= data1_drpd.drop(columns=['State'])
numerical_cols=numerical_df.columns
state_column = data1_drpd[["Unique_ID","State"]]

# Create numerical_df by excluding the 'State' column
numerical_df_original_vars= data1_drpd[["Preg_count", "BC1", "BC2", "BC3", "Blood Pressure", "Skin Thickness", "BMI", "GPF", "Age", "AQI", "Outcome", "Pregnant"]]
numerical_df_original_vars_cols=numerical_df_original_vars.columns


# After grouping the data we're left with 720 observations which is not a big dataset

# ### Outcome variable distribution

# In[164]:


outcome_counts = data1_drpd['Outcome'].value_counts()

print(outcome_counts)


# We can see that the outcome is not evenly distributed but it is not super heavy unbalanced

# ### Outliers handling

# In[165]:


# Identyfing possible outliers with Z-Score method
z_scores = np.abs((numerical_df - numerical_df.mean()) / numerical_df.std())
outliers_zscore = numerical_df[(z_scores > 3).any(axis=1)]
#outliers_zscore


# In[166]:


# Checking outcome variable in the outliers dataframe
outcome_counts = outliers_zscore['Outcome'].value_counts()

print(outcome_counts)


# In[197]:


# Removing outliers from the df
data1_drpd = numerical_df[~numerical_df.index.isin(outliers_zscore.index)]
data1_drpd = data1_drpd.merge(state_column, on='Unique_ID')


# The presence of outliers in the dataset is clear. These outliers may have been the result of errors in data entry. The removal of these outliers has the potential to improve the performance of our predictive model. In order to address this issue, we removed all outliers decected by the zscore method. In the scenario with more time available the outliers treatment should have been done manually and more carefully.

# ### Categorical variable handling

# In[168]:


# Encoding state variable using OHE
cat_col=['State']
data_dummies = pd.get_dummies(data1_drpd, columns=cat_col, drop_first=True)


# We use One-Hot-Encoding to deal with the categorical "state" variable in our dataset.

# # Visualisations

# We bring to action colours and charts to visualise how the data is distributed around outcome and other variables

# In[198]:


# Proportion of Positive Outcome across State
plt.figure(figsize=(18, 6))
sns.barplot(x='State', y='Outcome', data=data1_drpd, ci='sd')
plt.title('Proportion of Positive Outcome across State')
plt.show()


# States with higher bars like Texas or Alasca have a higher proportion of positive outcomes, indicating a higher incidence of the disease. On the other hand, states like Connecticut and Colorado with lower bars have a lower proportion of positive outcomes, suggesting a lower incidence of the disease.

# In[199]:


# Box Plot for Outcome vs. Preg_counts and Blood Pressure
plt.figure(figsize=(15, 8))
sns.boxplot(x='Preg_count', y='Blood Pressure', hue='Outcome', data=data1_drpd)
plt.title('Box Plot Outcome vs. Preg_counts and Blood Pressure')
plt.show()


# The boxes are slightly different between 'Outcome' groups for a specific No. of pregnancies, it indicates for a comples relationship between 'Blood Pressure', 'Preg_count', and the Outcome.

# In[172]:


# KDE Plot for BC1
plt.figure(figsize=(15, 6))
sns.kdeplot(data=data1_drpd, x='BC1', hue='Outcome', fill=True, common_norm=False)
plt.title('Density Estimation of BC1 with Hue for Outcome')
plt.show()


# From the curves on the KDE plot we can understand that the blood chemistry test has its importance in the presence of the disease in the group. Usually, the sick patients tend to have higher numbers for a blood chemistry test 1. The curves are visibly shifted, which means that the distribution of heathly patients is distributed around the value 100 of the BC1.

# In[200]:


# Heatmap for correlation
corr = numerical_df_original_vars.corr()
plt.figure(figsize=(13, 8))
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.title('Correlation Heatmap')
plt.show()


# We can see that BC1 and BMI are factors correlating with the outcome variable but the correlation is not extremly high. Other variables' correlation is quite low. Variables like BC3 or AQI do not look like strong indicators of the presence of the disease.

# # ML Approaches
# 

# ### Train/Test split
# 
# In order to proceed with ML approaches we need to first split the data into test, train and validation set. We'll need 3 groups in order to provide recomendation for the doctor on the unknown group of patents

# In[174]:


data_dummies.set_index('Unique_ID', inplace=True)

data_modeling =data_dummies
X = data_modeling.drop(columns=['Outcome'])
y = data_modeling['Outcome']

X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the remaining data into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("Validation set size:", X_val.shape)


# ## Random Forest
# 
# Random Forest, being an ensemble tecnique is a non-parametric model, which means it does not make strong assumptions about the data distribution. It's highly beneficial in medical datasets like ours, where the relationships between variables do not follow a specific mathematical pattern.
# 
# ### Model builing

# In[175]:


# Training
model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=50
)

model.fit(X_train, y_train)


# In[176]:


# Feature importance
feature_importances = model.feature_importances_

# Create a pandas DataFrame to store the feature importances
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importances
plt.figure(figsize=(15, 6))
plt.bar(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Features by Importance')


# In[177]:


# Model Evaluation
y_pred = model.predict(X_val)

# Accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[179]:


# Cross-Validation to get more robust evaluation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# In[ ]:


# Hyperparameters Tuning
param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}

grid_search = GridSearchCV(RandomForestClassifier(),param_grid=param_grid)
grid_search.fit(X_train, y_train)

# Best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)


# In[181]:


#  Model Interpretability (Partial Dependence Plots)

features_to_visualize = ['Preg_count', 'BC1', 'BMI', 'GPF', 'Age']

fig, ax = plt.subplots(figsize=(12, 8))
plot_partial_dependence(best_model, X_train, features_to_visualize, ax=ax, grid_resolution=50)
plt.suptitle('Partial Dependence Plots')
plt.subplots_adjust(top=0.9)
plt.show()


# We can observe that as 'BC1' increases, the risk score tends to increase as well, indicating a potential positive relationship between the chemistry blood test and the risk of disease incidence, similarly, for BMI and Age of patients. We can see however that for Preg_count, Genetic Predisposition Factor and Count of pregnancies, their values affect the predicted risk score a bit less.
# 
# 
# 
# 
# 
# 

# In[ ]:


# Making predictions on the validation set using the best model
y_pred = best_model.predict(X_val)

# Accuracy calculartion
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Disease presence')
plt.ylabel('True Disease presence')
plt.title('Confusion Matrix')
plt.show()


# In medical cases, especially when dealing with detecting diseases, recall is the most important metric. It represents the ability of the model to correctly identify patients with the disease. In the case of a random forest model the recall is 67% which is not the greatest result. 

# In[183]:


# Feature importances from the best model
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plot of feature importances
plt.figure(figsize=(15, 6))
plt.bar(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 15 Features by Importance')
plt.show()


# The feature importance analysis give us strong reasons to belive that the Blood Chemistry 1 results are the most influencial for predicting sick patients. We can see that the average and maximum value are playing the crucial role in the modeling. BMI and Age are on the other steps of the podium however their importance is not as high as BC1.

# ### Recomendations

# In[184]:


# Using best model we predict the probability estimates for each patient in the test set
patient_probabilities = best_model.predict_proba(X_test)

# Probability of the patient being sick
patient_risk_scores = patient_probabilities[:, 1]

# Coming up with classification based on a chosen threshold (0,5). If the risk score is greater than or equal to 0.5, the patient is classified as 'positive' (1), otherwise 'negative' (0)
patient_classifications = (patient_risk_scores >= 0.5).astype(int)

# Creating a separate DF with the results
results_df = pd.DataFrame({
    'Risk_Score': patient_risk_scores,
    'Classification': patient_classifications
}, index=X_test.index)

# Joining the DataFrames
merged_data = X_test.merge(results_df, left_index=True, right_index=True)

merged_data


# The model's predict_proba() method is used to predict the probability estimates for each patient. Then, the risk scores and classifications are calculated based on the probability estimates. It gives a clear information on the probability of the occurence of a disease of patients stored in the test dataframe with other variables.

# ## Boosted logistic regression
# 
# Another model we can try is a boosted logistic regression .As an ensemble method it combines multiple weak learners (logistic regression models) to create a strong learner. It can easily capture non-linear relationships between our features and the target variable, making it suitable for medical datasets where complex interactions and non-linearities may exist. It also has a great imbalanced Class Handling as i can give more emphasis to the minority class during model training which is beneficial in our case.
# 
# ### Model builing

# In[185]:


# Hyperparameter defeition
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 4, 5]
}

# Creating the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()

# Perform GridSearchCV for hyperparameter tuning using 5-fold cross-validation
grid_search = GridSearchCV(gb_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)


# In[186]:


best_model.fit(X_train, y_train)


# In[187]:


# Extract feature importances
feature_importances = best_model.feature_importances_

# Create a DataFrame to store feature importances with corresponding feature names
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(15, 8))
plt.bar(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15])
plt.title('Top 15 Features by Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In the Boosted Logistic Regression feature importance analysis, the results are a bit different then in RF approach. Similarly we observe an advantate of 1st Blood Chemistry test being the most important factor, then we have the patient's age and BMI as the indicators of a disease.

# In[188]:


# Model's performance

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Precision, Recall, and F1-Score (Classification Report)
class_report = classification_report(y_val, y_pred)
print("Classification Report:")
print(class_report)


# Assesing the model performance we can draw some conclusions:
# - The accuracy of the model is approximately 0.78. It represents the proportion of correctly predicted patients out of all predictions made by the model. In this case, we can assume it's a great performance
# - Out of all the patients predicted as sick by the model, around 69% of them are actually true positive cases, which is not promising but alright.
# - From all actual sick patients in the validation set, the model was able to correctly identify around 61% of them. Recall, being the most important metric to judge the model performance is not very high which leaves place for improvement.
# 
# Overall, the model shows reasonable performance with good accuracy, but there is room for improvement in correctly identifying the positive class (class 1) which is the most important in our case.

# ### Recommendations
# 
# In Boosted Logistic Regression we also provide recomendation to the doctor about patients, which we obtain by the predict_proba() function.

# In[189]:


# Using best model we predict the probability estimates for each patient in the test set
patient_probabilities_log_reg = best_model.predict_proba(X_test)

# Probability of the patient being sick
patient_risk_scores_log_reg = patient_probabilities_log_reg[:, 1]

# We can come up with classification based on a chosen threshold (0,5). If the risk score is greater than or equal to 0.5, the patient is classified as 'positive' (1), otherwise 'negative' (0)
patient_classifications_log_reg = (patient_risk_scores_log_reg >= 0.5).astype(int)

# Creating a separate DF with the results
results_log_reg_df = pd.DataFrame({
    'Risk_Score': patient_risk_scores,
    'Classification': patient_classifications_log_reg
}, index=X_test.index)

# Joining the DataFrames 
merged_datas_log_reg = X_test.merge(results_log_reg_df, left_index=True, right_index=True)

merged_datas_log_reg


# # Conclusions

# Overall in the performed analysis we focused strongly on the data preprocessing and preparation, creating additional variables which turned out to have an interesting influence on the modeling part. It required some effort to deal with inconsistencies in the data and then draw first conclusions based on short EDA.\
# 
# In the modeling part we came up with 2 models being Random Forest and Boosted Logistic Regression as they offered a wide range of advantages for our case. As a result we got conclusions on the importance of factors causing a disease in the patients' examined group, which for both models turned out to be Blood Chemistry 1st tests. It means that a high value of this test could be a red flag for the doctor.\
# 
# Moreover, we provided the doctor with a list of patients with probabilies of the disease presence so that she can draw conclusions and help them on time.
