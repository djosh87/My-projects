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
