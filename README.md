# Auto-Insurance-Criminal-Deception

## I. Define the problem:
* The goal of this project is to build a model that identifies genuine auto claims from fraudulent ones. The challenge behind fraud detection in machine learning is that frauds are far less common as compared to legit insurance claims. This type of problems is known as imbalanced class classification.
* Frauds are unethical practices and are losses to the company.This model will help the auto insurance companies to pay the customers who are really affected by the accident and to screen the customers who are falsesome.
* If it is identified the claim submitted is not genuine it will go for manual inspection. And the person can put up proofs to conclude if his claim is genuine before the decision is made. By building this fraud classifier, one can cut losses for the insurance company.

**Environment or libraries used**

* Google Colab
* Python 3
* Numpy
* Pandas
* Sklearn
* Matplotlib
* Keras

##II. DISCOVER 
A .csv file containing the dataset is used for this project. The various attributes of the dataset are explianed further.

ATTRIBUTES OF DATASET:
1.	months_as_customer- It denotes number of months for which the customer is associated with the insurance company.
2.	age- It denotes person’s age.
3.	policy_number- denotes the insurance policy number.
4.	policy_bind_date- Start date of the policy.
5.	policy_state- The state where the policy is registered.
6.	policy_csl- Combined single limits. How much of the bodily injury will be covered from the total damage.
Example- 250/500- If injury is total 500$ then only 250$ will be covered from your policy.
7.	policy_deductable- The amount paid out of pocket by the policy holder before an insurance provider will pay any expenses.
8.	policy_annual_premium- The yearly premium for the policy.
9.	umbrella_limit- An umbrella insurance policy is extra liability insurance coverage that goes beyond the limits of the insured’s homeowners, auto or watercraft insurance. It provides an additional layer of security to other people’s property or injuries caused to others in an accident.
10.	insured_zip- The zip code where the policy is registered.
11.	insured_sex- It denotes the person’s gender.
12.	insured_education_level- The highest educational qualification of the policy holder.
13.	insured_occupation- The occupation of the policy holder.
14.	insured_hobbies- The hobbies of the policy holder.
15.	insured_relationship- Dependents on the policy holder.
16.	capital-gain- It denotes the minority gains by the person.
17.	capital-loss- It denotes the monitory loss by the person.
18.	incident_date- The date when the incident happened.
19.	incident_type- The type of incident.
20.	collision_type- The type of collision that took place.
21.	incident_severity- The severity of the incident.
22.	authorities_contacted- Which authority was contacted.
23.	incident_state- The state in which the incident took place.
24.	incident_city- The city in which the incident took place.
25.	incident_location- Street number in hich your acc
26.	incident_hour- At what time of the day did your acciden took place.
27.	number_of_vehicles_involved- how many vehocles involved in the accident
28.	property_damaged- Where propoety damage to some other person was one
29.	bodily_injuries- How many bodily injuries you suffered being a part of the incident
30.	witnesses- How many witnesses present at the scene.
31.	police_report- Whether the police report was registered or not
32.	total_claim- Combination of the bodily injuires, property damage and vehicle damage
33.	injury_claim- Claim for the bodily injuries
34.	property_claim- Claim for the property damage
35.	vehicle_claim- Claim for the vehicle.
36.	auto_make- The make of the vehicle
37.	auto_model- The model of the vehicle
38.	auto-year- In which year you purchase your vehicle
39.	fraud_reported- If the fraud was recorded.


## II. DISCOVER
**Data pre-processing**

1. Dropping the columns which are not necesssry for  prediction which includes the following: 'policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year'
2. Check for missing values
3. Missing values are categorical in nature, so we use Categorical Imputer to fill in the missing values.
4. Extract the categorical columns and numerical columns from the dataset.
5. Encode categorical columns to convert them into numerical form.
6. Concatenate the categorical and numerical dataframes  to get the final dataset
7. Seperate the target column('fraud reported') from the feautures. 
8. Checking the target column we can understand that the 'fraud reported' is an imbalanced column with 75.3% fraud unreported and only 24.7% frauds registered.

**Exploratory Data analysis**
1. From the scatter plot it can be concluded that most of the fraud cases are done by the customers new to the company and that too comparatively younger ones. There is a high and positive correlation between age of the driver and the months they have been engaged to the insurance company.
![Scatterplot.png](https://github.com/Jimisha18/Auto-Insurance-Criminal-Deception/blob/main/Images/Scatterplot.png)

2. From the heatmap there is high correlation recorded between total claim amount, injury claim, vehicle claim and property claim as total claim is the sum of all others. 
![Heat map.png](https://github.com/Jimisha18/Auto-Insurance-Criminal-Deception/blob/main/Images/Heat%20map.png)

3. Drop age and total claim amount columns.

## III. DEVELOP
1. Splitting the dataset into training and testing data
2. Scaling the numerical dataframe in the train data and check for any missing values post scaling
3. Merging the scaled numerical data with the train dataset.
4. Build SVM classifier as the baseline model. 

### MODEL-1 SUPPORT VECTOR MACHINE
```
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sv_classifier=SVC()
y_pred_svc = sv_classifier.fit(X_train, y_train)
y_predict_svc=sv_classifier.predict(X_test)
final_accuracy_svc=accuracy_score(y_predict_svc,y_test)
print("The test-accuracy of SVM model is ", final_accuracy_svc*100,"%")
```
The test-accuracy of SVM model is  72.8 %

5. Hyperparameter tunning for baseline SVM model
```
from sklearn.model_selection import GridSearchCV
param_grid = {"kernel": ['rbf','sigmoid'],
             "C":[0.1,0.5,1.0],
             "random_state":[0,100,200,300]}
grid = GridSearchCV(estimator=sv_classifier, param_grid=param_grid, cv=5,  verbose=3)
grid.fit(X_train, y_train)
```

6. Fit the baseline model by testing the accuracy of SVM for best value of C
```
from sklearn.metrics import accuracy_score
sv_classifier = SVC(kernel='rbf',C=0.1, random_state=0)
sv_classifier.fit(X_train, y_train)
y_predict_svc=sv_classifier.predict(X_test)
final_accuracy_svc=accuracy_score(y_predict,y_test)
print("The test-accuracy of tunned-SVM model for best C value is ", final_accuracy_svc*100,"%")
```
The classification by SVM model has no major improvement using the tunned parameter c=0.1 so let us try XG-Boost model.

7. Build XG Boost model

### MODEL-2 XG Boost
```
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb=XGBClassifier()
y_pred_xgb = xgb.fit(X_train,y_train)
y_predict_xgb=xgb.predict(X_test)
final_accuracy_xgb=accuracy_score(y_predict_xgb,y_test)
print("The test-accuracy of XG Boost model is ", final_accuracy_xgb*100,"%")
```
The test-accuracy of XG Boost model is  77.2 %

8. Hyperparameter tunning for XG-Boost model
```
param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
             "max_depth": range(2, 10, 1)}
             
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,  verbose=3,n_jobs=-1)
grid.fit(X_train, y_train)
```
For XG-Boost classifier applying Grid search CV we get the tunned parameters such as the max_depth=2 and n_estimators=10.

9. Fit the XGBoost model using the tunned parameters.
```
from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=10, max_depth= 2)
y_pred_xgb = xgb.fit(X_train, y_train)
y_predict_xgb=xgb.predict(X_test)
final_accuracy=accuracy_score(y_predict_xgb,y_test)
print("The test-accuracy of tunned XG-Boost model is ", final_accuracy*100,"%")
```

The test-accuracy of tunned XG-Boost model is  80.0 %

Using the tunned parameters we are elevating the accuracy of the classifier to 80%. Which was not the case in SVM. 

Hence, XG-Boost classifier performs far better than the baseline SVM model in classifying the auto-insurance fraud.

