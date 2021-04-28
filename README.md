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


## II. 
