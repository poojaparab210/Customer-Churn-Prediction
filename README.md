# Customer Churn Prediction



Hi! This is a Data Science and Machine Learning project I built to understand why customers leave a service and how businesses can identify customers who may be likely to leave.



The project uses customer data to predict churn and provides an interactive dashboard where users can explore customer behaviour, check churn predictions, and view suggestions for customer retention.



\## What does this project do?



\- \*\*Customer Churn Prediction:\*\* Predicts the likelihood of a customer leaving a service.

\- \*\*Dashboard:\*\* Shows an overview of customer churn and helps explore patterns in the data.

\- \*\*Customer Risk Analysis:\*\* Groups customers into low-, medium-, and high-risk categories based on their predicted churn probability.

\- \*\*Retention Recommendations:\*\* Provides suggestions based on customer details and their risk category.

\- \*\*Feature Importance:\*\* Helps explore which features influence the model's predictions.

\- \*\*Batch Prediction:\*\* Allows users to upload a CSV file containing multiple customers and download the prediction results.



\## Tools and Technologies



I used the following tools to build this project:



\- Python

\- Pandas and NumPy for data handling

\- Matplotlib and Seaborn for data visualization

\- Scikit-learn for machine learning

\- Plotly for interactive charts

\- Streamlit for the web application

\- Joblib for saving and loading the trained model



\## Dataset



This project uses the Telco Customer Churn dataset. It contains information about customers, the services they use, their contracts, billing details, and whether they left the service.



The target column is `Churn`, which indicates whether a customer has left.



\## How I built the project



I followed these steps while developing the project:



1\. Cleaned the dataset and prepared it for analysis.

2\. Performed exploratory data analysis (EDA) to understand customer churn patterns.

3\. Trained and compared Logistic Regression and Random Forest models.

4\. Evaluated the models using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

5\. Tuned the classification threshold and selected Logistic Regression for the production prediction pipeline.

6\. Created customer risk categories and rule-based retention recommendations.

7\. Built a Streamlit application to bring the results together in one place.



\## Project Structure



```text

CustomerChurnProject/

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

├── data/

├── src/

└── outputs/

&#x20;   ├── cleaned\_churn.csv

&#x20;   ├── eda\_charts/

&#x20;   ├── feature\_importance/

&#x20;   ├── model\_evaluation/

&#x20;   ├── models/

&#x20;   ├── customer\_risk/

&#x20;   └── retention/

```



\## How to run the project



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd CustomerChurnProject

```



\### 2. Create a virtual environment



On Windows PowerShell, run:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install the required packages



```powershell

python -m pip install -r requirements.txt

```



\### 4. Start the application



```powershell

python -m streamlit run app.py

```



The application should open in your browser.



\## What I learned



While working on this project, I practised data cleaning, exploratory data analysis, model training, model evaluation, classification threshold tuning, and building an interactive application with Streamlit. It also helped me understand how machine learning predictions can be used to explore customer retention opportunities.



\## Future Improvements



I would like to improve this project further by deploying it online, testing it with new customer data, and evaluating whether the retention recommendations lead to better business outcomes.



\## Author



\*\*Pooja Parab\*\*  

B.Sc. Computer Science  

Aspiring Data Science and Machine Learning Professional



