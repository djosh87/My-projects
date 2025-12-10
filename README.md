# 📊 Data Science & Machine Learning Projects
This repository contains multiple data analysis and predictive modeling projects developed using Python and Machine Learning techniques.
Objective of this project is analyzing the data to get a fair idea about the demand of customers which will help Austo Motor LMT in enhancing their customer experience.
## 🚗 Project 1: Automobile Customer Demand Analysis

## Tools & Technologies
- Python (pandas, numpy, scikit-learn)
- Jupyter Notebook
- Matplotlib / Seaborn
- ## Dataset
- Synthetic telecom customer dataset with ~5,000 records and 15+ features
  Data Over View
  - Importing required   libraries 
- Loading the data
- Validating structure of the data
- Validating types of the data
- Checking for and treat (if needed) missing values
- Validating statistical summary
Criteria check
Univariate Analysis
- Explored all the variables (categorical and numerical) in the data
- Checked for and treat (if needed) outliers
Bivariate Analysis
 Explored the relationship between all numerical variables
- Explored the correlation between all numerical variables
- Explored the relationship between categorical vs numerical variables

  Insights :
  Salary the vital thing to consider for the sales
• Partner salary is inversely affecting to sales i.e.as Partner salary increasing sale is decreasing
• Total salary is inversely affecting to sales i.e.as Total salary increasing sale is decreasing
• Customer is more interested in buying cars which are low in cost
• With the higher price, percentage of sale is reducing
• Men prefers to buy a car more than women.
• Women prefer to buy SUVs more than men
• Men prefers more sedan and hatch back than women
• Very less women prefers hatchbacks
• Sale is more with the customers who are salaried
• Salaried persons prefer sedan more than any other model
• Hatchback is the second preferred model for both salaried and businesspeople
• SUV is last preferred option by businesspeople
• Married persons prefer to buy car more than a person who is single
• Postgraduates prefers to buy car more than graduates
• Personal loan entity is not affecting the sales. sale is almost same for the customer who takes personal loan and who does not take personal loan
• Customers who are having house loans are not preferring buy car.
• Customers who are not having house loans are preferring to buy a car.
• Customer whose partners are working prefer to buy car more than the customers whose partners are not working
• Make is the important aspect for the car sale.
• More customers prefer to buy sedan make
• Hatchback is the second most preferred
• SUV the last preferred make

📈 Project 2: Predictive Analytics Project
An OTT service provider  offers a wide variety of content (movies, web shows, etc.) for its users. They want to determine the driver variables for first-day content viewership so that they can take necessary measures to improve the viewership of the content on their platform. Some of the reasons for the decline in viewership of content would be the decline in the number of people coming to the platform, decreased marketing spend, content timing clashes, weekends and holidays, etc. They have hired you as a Data Scientist, shared the data of the current content in their platform, and asked you to analyze the data and come up with a linear regression model to determine the driving factors for first-day viewership.

📂 Dataset
The dataset includes structured numerical and categorical features used for prediction and analysis.

🛠 Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook

🤖 Machine Learning Model
Model building - Linear Regression
- Built the model and comment on the model statistics
- Displayed model coefficients with column names

  ✅ Key Outcomes
  Exploratory Data Analysis done
  Duplicate value checked
- Missing value treatment done
- Outlier treatment done
- Feature engineering wherever required is done
- Data preparation for modeling is done
  The model is able to explain ~79% of the variation in the data and within 9.1% of the views_content on the test data, which is good
- This indicates that the model is good for prediction as well as inference purposes
2. The model indicates that the most significant predictors of the views_content are the following:
•
Visitors
•
Major sports event
•
Views_trailer
•
Dayofweek
•
Season
(The P value of above predictors are less than 0.05 in our final model)
3. If the number of visitors, visiting the platform increases by one unit, then views_content increases by 0.1291 units, all other variables held constant
4. Major_sports_event is inversely correlated with views of content i.e. on content release day presence of major sports event will decrease the content_vies by 0.06 unit
6. The day on which content released is important factor to decide the views_content, content released on Friday having more viewer ship
7. Season in which content released also matters to decide the views_content
8. The train and test RMSE and MAE are low and comparable. So, our model is not suffering from overfitting
9. The MAPE on the test set suggests we can predict within 9.1% of the views_content
10.Hence, we can conclude the model olsmodel_final is good for prediction as well as inference purposes

 Project 3 - INN Hotels
 
### 📌 Objective
The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. INN Hotels Group has a chain of hotels in Portugal, they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions. You as a data scientist have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.

📂 Dataset
Data contains hotel related data like booking id,no of adults,no of kids,no of week end night ,week day nights,booking status 

🛠 Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook
🤖 Machine Learning Model
  - Logistic Regression (statsmodels)
    - Decision Tree Classifier (sklearn)
   
  Addressed Below :
  Exploratory Data Analysis
- Univariate analysis
- Bivariate analysis
- Used appropriate visualizations to identify the patterns and insights
- Key meaningful observations on individual variables and the relationship between variables
  - Missing Value Treatment (with rationale if needed)
- Outlier Detection and Treatment (with rationale if needed)
- Feature Engineering (with rationale if needed)
- Data Scaling (with rationale if needed)
- Train-test split

  Model Performance Improvement
Logistic Regression (deal with multicollinearity, remove high p-value variables, determine optimal threshold using ROC curve)
There are different ways of detecting (or testing for) multicollinearity. One such way is using the Variation Inflation Factor (VIF).

Variance Inflation factor: Variance inflation factors measure the inflation in the variances of the regression coefficients estimates due to collinearities that exist among the predictors. It is a measure of how much the variance of the estimated regression coefficient βk is "inflated" by the existence of correlation among the predictor variables in the model.

General Rule of thumb:

If VIF is 1 then there is no correlation among the kth predictor and the remaining predictor variables, and hence the variance of βk is not inflated at all.

If VIF exceeds 5, we say there is moderate multicollinearity.

If VIF is equal or exceeding 10, it shows signs of high multi-collinearity.

The purpose of the analysis should dictate which threshold to use.

Cost Complexity Pruning
The DecisionTreeClassifier provides parameters such as min_samples_leaf and max_depth to prevent a tree from overfitting. Cost complexity pruning provides another option to control the size of a tree. In DecisionTreeClassifier, this pruning technique is parameterized by the cost complexity parameter, ccp_alpha. Greater values of ccp_alpha increase the number of nodes pruned. Here we only show the effect of ccp_alpha on regularizing the trees and how to choose a ccp_alpha based on validation scores.

Total impurity of leaves vs effective alphas of pruned tree
Minimal cost complexity pruning recursively finds the node with the "weakest link". The weakest link is characterized by an effective alpha, where the nodes with the smallest effective alpha are pruned first. To get an idea of what values of ccp_alpha could be appropriate, scikit-learn provides DecisionTreeClassifier.cost_complexity_pruning_path that returns the effective alphas and the corresponding total leaf impurities at each step of the pruning process. As alpha increases,more of the tree is pruned, which increases the total impurity of its leaves.

Actionable Insights and Recommendations
Insights

Most of the customer prefers to spend 2 weekend nights in the hotel

Most customers booked hotel well in advance and data is telling cancellation is less for these customers.

Most customers booked hotel well in advance and data is telling cancellation is less for these customers.

cancellation rate is less for the repeated guests

There are dynamic prices observed with online bookings

As special requirements increase cancellation rate is decreasing and vice versa i.e. customer with more special requirement confirms the booking and will not cancel.

0 special requirements more cancellation rate is observed.

As price is increasing cancellation rate is increasing
Recommendations:
Decision tree is the best model to predict the booking cancellations

Hotel can offer some discount price and offers to the customer who book in advance to encourage

Repeated guests are very important to the business as they spread the brand value and hotel should concentrate on these customers

Hotel needs to work on lead time, average price per room and number of special requests to keep up brand equity.

Customers can be notified of lead time, and they can be asked for any special requests before stay.

Special amenities list can be provided after booking to reduce chances of cancellation.
Depending on the Lead time, average price per room can be adjusted to attract more customers to the hotel.

Depending on the number of special requests, hotel can adjust the average price per room to maintain resources and brand equity.

Hotel can come up with policies on cancellations or refund based on lead time.

Project 4 : EasyVisa

Objective  :  The increasing number of applicants every year calls for a Machine Learning based solution that can help in shortlisting the candidates having higher chances of VISA approval. OFLC has hired the firm EasyVisa for data-driven solutions. You as a data scientist at EasyVisa have to analyze the data provided and, with the help of a classification model:

Facilitate the process of visa approvals.
Recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the drivers that significantly influence the case status.

📂 Dataset
The data contains the different attributes of the employee and the employer. The detailed data dictionary is given below.

case_id: ID of each visa application
continent: Information of continent the employee
education_of_employee: Information of education of the employee
has_job_experience: Does the employee have any job experience? Y= Yes; N = No
requires_job_training: Does the employee require any job training? Y = Yes; N = No
no_of_employees: Number of employees in the employer's company
yr_of_estab: Year in which the employer's company was established
region_of_employment: Information of foreign worker's intended region of employment in the US.
prevailing_wage: Average wage paid to similarly employed workers in a specific occupation in the area of intended employment. The purpose of the prevailing wage is to ensure that the foreign worker is not underpaid compared to other workers offering the same or similar service in the same area of employment.
unit_of_wage: Unit of prevailing wage. Values include Hourly, Weekly, Monthly, and Yearly.
full_time_position: Is the position of work full-time? Y = Full-Time Position; N = Part-Time Position
case_status: Flag indicating if the Visa was certified or denied

Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook

🤖 Machine Learning Model
decision trees, bagging and boosting methods

Addressed Below :

Exploratory Data Analysis
- Univariate analysis
- Bivariate analysis
- Used appropriate visualizations to identify the patterns and insights
- Key meaningful observations on individual variables and the relationship between variables
- - Prepare the data for analysis
- Feature Engineering
- Missing value Treatment
- Outlier Treatment

  Business Insights and Recommendations
  Continent affecting visa approval
• Maximum employees who are certified for visa are from Europe
• second highest employees who are certified for visa are from Africa
• People from Oceania are least preferred for visa
Education level-affecting visa approval
• People are who are holding doctorate degree are considered first for visa certification
• Next preferred people are the people who holds master’s degree
• Next preferred are the people who holds degree
• High school qualified people are least preferred for the visa certification
Prior Job experience affecting visa approval
• Most of the employees are having prior job experience and preferred more for visa approval
Job training affecting visa approval
• Most of the employees does not need any job training and these people are more preferred for visa as they are cost effective

Other insights into the data
• Full time working employees are preferred more for visa approval
• 67% employee's visa is certified
• 33% employee's visa is not certified
• prevailing wages is higher in Midwest and Island
• The median prevailing wage for the certified applications is slightly higher as compared to denied applications.

Recommendations
• Training and test performance of ada boost data with under sampled data is comparable i.e. 81%
• we can conclude that ada boost with under sampled data model is more generalised and we can go with
this model to predict the visa approval process

Project 5 : Unsupervised_AllLife

Objective  :To identify different segments in the existing customers, based on their spending patterns as well as past interaction with the bank, using clustering algorithms, and provide recommendations to the bank on how to better market to and service these customers.

 Data Description
The data provided is of various customers of a bank and their financial attributes like credit limit, the total number of credit cards the customer has, and different channels through which customers have contacted the bank for any queries (including visiting the bank, online, and through a call center).

Data Dictionary
Sl_No: Primary key of the records
Customer Key: Customer identification number
Average Credit Limit: Average credit limit of each customer for all credit cards
Total credit cards: Total number of credit cards possessed by the customer
Total visits bank: Total number of visits that the customer made (yearly) personally to the bank
Total visits online: Total number of visits or online logins made by the customer (yearly)
Total calls made: Total number of calls made by the customer to the bank or its customer service department (yearly)

Model Building :
K-means Clustering : Ploting the Elbow curve 
- Checking Silhouette Scores
- Figuring out the appropriate number of clusters
- Clustering Profiling
Hierarchical Clustering
- Ploting dendrograms for each linkage method
- Checking cophenetic correlation for each linkage method
- Figuring out the appropriate number of clusters
- Cluster Profiling

  Addressed Below :
  Exploratory Data Analysis
- Univariate analysis
- Bivariate analysis
- Used appropriate visualizations to identify the patterns and insights
- Key meaningful observations on individual variables and the relationship between variables
- Data Preprocessing
- Missing Value Treatment (with rationale if needed)
- Outlier Detection and Treatment (with rationale if needed)
- Feature Engineering (with rationale if needed)
- Data Scaling (with rationale if needed)

  Insights and Recommendations
- As there are less data here, both clusters took same time to execute
-Both clustering methods produced 3 clusters
-K means has 386 observations in cluster 0 ,224 observations in cluster 1 and 50 observations in cluster 2
-Cluster 2 is having customers who are having more credit card limit, maximum credit cards and k means segregated this cluster with 50 observations.
-AllLife bank needs to focus more on cluster 2 as it is having highest credit limits and credit cards, and this group could be at more risk if they default
-Cluster 1 is having 224 observations, indicates it contains customers who are having less credit card limits and they show low risk to the bank
-Cluster 0 is having 386 observations, indicates it contains customers who are having average credit card limit, and they show moderate risk to the bank
-Hierarchical cluster has 386 observations in cluster 0 ,50 observations in cluster 1 and 224 observations in cluster 2.
-Cluster 2 is having customers who are having more credit card limit, maximum credit cards and k means segregated this cluster with 224 observations.
-Both methods produced same no of clusters and observations, but k means segregated less customers in cluster 2 which can show high risk to the bank and as it is having less no of customers, it is more cost effective and easily manageable and hierarchical cluster methodology segregated more customers in cluster 2 which can show high risk to the bank and as it is having more no of customers, it can become expensive and difficult to manage large no of customers who can become high risk to the bank if they default
-We can conclude that AllLife bank can go with K means cluster methodology


Project 6 : AI

# 🌍 AI-Based Multilingual Speech & Text Processing System

This project is an AI-powered automation system that performs **multilingual text translation** and **speech-to-text transcription** using the **SarvamAI API** and Python. It supports **Indian regional languages** such as **Gujarati and Hindi**, enabling real-world applications in accessibility, customer support, and voice-enabled systems.

---

## 🚀 Project Features

✅ Automatic **Text Translation (English → Indian Languages)**  
✅ **Speech-to-Text Transcription** from audio and video files  
✅ Supports **Gujarati (gu-IN)** and **Hindi (hi-IN)**  
✅ **AI-based NLP & Speech Recognition**  
✅ **Audio extraction from video files** using MoviePy  
✅ End-to-end **Python automation with API integration**

---

## 🧠 Technologies Used

- Python  
- SarvamAI API  
- Natural Language Processing (NLP)  
- Speech Recognition  
- MoviePy (Audio extraction from video)  
- REST API Integration  
- Jupyter Notebook  

---

## 📌 Use Cases

- AI Voice Assistants  
- Multilingual Customer Support Automation  
- Regional Language Content Generation  
- Accessibility Tools for Speech-Impaired Users  
- Video-to-Text Transcription Systems  

---

## 🔹 Module 1: Text Translation (Text → Text)

This module takes English text as input and translates it into Indian regional languages using AI.

### ✅ Example Use Case:
- Input: `"Hi, My Name is Vinayak."`
- Output: Translated Gujarati Text

### ✅ Key Capabilities:
- Auto language detection
- Supports multiple Indian languages
- Gender-based voice customization

---

## 🔹 Module 2: Speech-to-Text Transcription (Audio → Text)

This module converts spoken language from **audio or video files** into readable text using deep learning-based speech models.

### ✅ Supported Formats:
- WAV
- MP4 (after audio extraction)

## 🛠 Installation & Setup

### 1️⃣ Install Required Libraries
```bash
pip install sarvamai moviepy

