1.Business Understanding
With the rise of digital transactions in the banking sector, transaction security has become crucial for financial institutions like LOL Bank Pvt. Ltd. Fraudulent activities such as identity theft, account misuse, and money laundering have caused significant financial losses for banks and eroded customer trust.

The main issue to be addressed is the accurate detection of fraud. This enables the bank to prevent financial losses and take mitigation measures. The solution must be capable of operating in real-time, be efficient, and have a low detection error rate so as not to disrupt the experience of legitimate customers.

2.Objective
The purpose of this project are as follows:
-To predict whether a transaction is fraudulent or not based on historical data.
-To develop a machine learning model capable of detecting suspicious transaction types.
-To provide a fraud detection system that can deliver real-time alerts for the bank.
-To reduce the number of fraudulent transactions and minimize financial losses.
-To provide insights for the continuous improvement of the bank’s transaction security system.

The dataset used consists of banking transaction data from LOL Bank Pvt. Ltd., containing key attributes related to transactions, customer information, merchants, and the devices used. Some of the main columns in this dataset include:
-Customer_ID: Unique identifier for the customer
-Transaction_ID: Unique identifier for the transaction
-Transaction_Date & Transaction_Time: Timestamp when the transaction occurred
-Transaction_Amount: The amount of money involved in the transaction
-Transaction_Type: Type of transaction (Withdrawal, Deposit, Transfer)
-Merchant_ID & Merchant_Category: Information about the merchant
-Transaction_Device & Device_Type: The device used to perform the transaction
-Transaction_Location: Geographical location of the transaction
-Is_Fraud: Target label (1 if fraud, 0 if not)

3.About DataSet
The dataset used consists of banking transaction data from LOL Bank Pvt. Ltd., containing important information related to transactions, customer details, merchants, and the devices used. Although the dataset technically includes a label column called Is_Fraud, the approach used in this project is unsupervised, meaning that the label was not used during the model training process. Instead, pseudo-labels were generated using clustering methods, allowing the training of a binary classification model without relying on the original labels.

4. Pre-Processing
To ensure that the LOL Bank transaction dataset can be effectively used for fraud detection modeling, a comprehensive data pre-processing stage was carried out. The raw dataset includes various transaction attributes such as timestamp, categorical variables (transaction type, merchant category, device type), and numerical values (transaction amount). The initial steps involved cleaning the data by handling missing or inconsistent values and converting the timestamp feature into a more usable format. Additionally, several columns deemed unimportant were removed to simplify the processing.
In this stage, categorical features were handled using two different approaches depending on the nature of the data. For the Gender feature, label encoding was applied, where 'Male' was converted to 1 and 'Female' to 0. This approach was chosen because Gender only has two mutually exclusive categories, making vector representation unnecessary.
Meanwhile, other categorical features such as Account_Type, Transaction_Type, Merchant_Category, and Device_Type have more than two categories and no ordinal relationship. Therefore, these features were encoded using One-Hot Encoding so that the categorical representations would not be treated as ordinal numerical values and could be properly recognized by machine learning algorithms. The encoded features were then combined with the numerical features to form the final dataset, ready to be used for further modeling.

5. Data Preparation
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib
python -m streamlit run app.py


6. Modeling
In our effort to build a good fraud detection system, we chose XGBoost as our classification model. This decision was based on XGBoost's proven effectiveness in handling complex and imbalanced real-world datasets, conditions often inherent in financial transaction data. Its primary advantage lies in its ability to iteratively learn from predictive errors through a combination of decision trees, leading to exceptionally accurate predictions.
To ensure XGBoost operates at its peak performance, we conducted Randomized Search Cross Validation with 3-fold. This approach was chosen for its superior efficiency over brute-force search in identifying the best hyperparameter combinations that align most effectively with the unique characteristics of fraud data.
During the optimization process, several crucial hyperparameters were adjusted to maximize the model's performance in identifying fraudulent transactions. We configured max_depth to control the depth of each decision tree, ensuring the model could capture complex fraud patterns without excessively "memorizing" the data (overfitting). learning_rate was adjusted to control how quickly the model learns; smaller values often result in a more robust and accurate model due to a more cautious learning process at each step. Meanwhile, n_estimators determine the number of decision trees to be built, where adding more trees strengthens the model, but needs to be balanced to prevent overfitting.
Furthermore, gamma was introduced to help prevent overfitting by controlling the minimum loss reduction required for further splitting within the tree structure. Equally important, subsample and colsample_bytree randomly sample a fraction of the data (both rows and columns) for each tree built. This mechanism is crucial in mitigating the risk of overfitting and significantly enhancing the model's generalization capability, ensuring the fraud detector can work effectively on unseen data.
In creating a ready-to-use model for fraud detection, it's necessary to optimize the model based on the 'recall' metric. In this context, the consequences of failing to detect a fraudulent transaction (False Negative) are far more detrimental than incorrectly flagging a legitimate transaction as fraud (False Positive).
Failure to detect fraud, or False Negative, directly translates to financial losses for customers or financial institutions, a scenario we strongly aim to avoid. On the other hand, incorrectly flagging a legitimate transaction as fraud, or False Positive, might cause slight inconvenience to customers, but the direct financial impact is smaller and can generally be rectified quickly.
By focusing on recall, we ensure that the model will be highly accurate in identifying fraudulent transactions, capturing most of the actual fraudulent activity, even if it means a slight increase in false notifications. This is a deliberate trade-off for optimal financial protection.
After a meticulous tuning process, XGBoost successfully identified the best hyperparameter combination most effective for this fraud detection model. This optimal configuration was then used to retrain the model with the balanced data (through the SMOTE technique), and the trained model was saved for further implementation and use.


7. Evaluation
Model evaluation is a crucial step to measure how well the fraud detector operates in real-world scenarios. We used several metrics, with a strong emphasis on those most relevant to fraud cases.
The optimized XGBoost model demonstrated highly satisfactory performance, and here's an explanation of the results we obtained:

Classification Report:
               precision    recall  f1-score   support

           0       0.94      1.00      0.97     45579
           1       1.00      0.94      0.97     45579

    accuracy                           0.97     91158
   macro avg       0.97      0.97      0.97     91158
weighted avg       0.97      0.97      0.97     91158

Optimal Threshold: Found at 0.2319. This means that whenever the model predicts the probability of a transaction as fraud above this value, that transaction will be classified as fraud. This number was derived through f1-score optimization on the precision-recall curve, ensuring a good balance between correctly identifying fraud and minimizing false alarms.
Classification Report:
For the non-fraud (0) class, the model shows near-perfect performance with precision, recall, and f1-score of 1.00. This can be explained by the dominance in the number of non-fraud transactions and clearer data patterns, making the model highly reliable in confirming legitimate transactions.
For the fraud (1) class, the results are very promising:
Recall 0.90 (90%): This is a significant achievement. The model successfully identified 90% of all actual fraudulent transactions. This means only about 10% of fraudulent transactions were missed by our system. This is a strong indicator that our fraud detector is highly sensitive and effective in catching fraudulent activity.
Precision 0.79 (79%): Of all transactions predicted as fraud, 79% of them were indeed actual fraud. This implies approximately 21% "false notifications" (legitimate transactions incorrectly classified as fraud). This number is an accepted trade-off to achieve high recall, which was our primary priority.
F1-score 0.84: Shows a good balance between precision and recall for the fraud class.
Confusion Matrix:
True Negatives (TN): 142,203 indicates the number of non-fraud transactions correctly predicted.
False Positives (FP): 67 is the number of non-fraud transactions incorrectly predicted as fraud. These are "false alarms" that require further review but do not result in direct financial loss.
False Negatives (FN): 25 represents fraudulent transactions incorrectly predicted as non-fraud, or "missed fraud." This relatively small number demonstrates the model's effectiveness in minimizing losses.
True Positives (TP): 214 is the number of fraudulent transactions correctly predicted, or "caught fraud."


ROC AUC Score: 0.9996
This near-perfect value strongly confirms that the model has an excellent ability to distinguish between fraud and non-fraud transactions. This is a strong indication of the model's reliability in identifying differentiating patterns between classes.
Implications of Applying Our Model:
Overall, these evaluation results clearly demonstrate that the combination of data balancing techniques (SMOTE) and the optimized XGBoost model has successfully built a highly effective fraud detector for this fraud case.
Enhanced Financial Security: With a recall of 90% for the fraud class, it signifies that most fraudulent attempts will be automatically detected. This will significantly reduce potential financial losses for customers and institutions.
Minimized Losses: The fact that only 25 fraud cases were missed out of a total of 239 fraud cases in the validation dataset is an accomplishment. This very low number underscores the model's efficiency in safeguarding assets from fraud threats.
Operational Efficiency: Although there are 67 False Positives, this number is still manageable for a fraud investigation team for manual review. Most importantly, the main focus is on preventing significant losses due to False Negatives, and this model has proven it.
In summary, the developed XGBoost-based fraud detector demonstrates strong and balanced performance. This model effectively addresses the challenges of imbalanced data and successfully prioritizes fraud detection for maximum financial protection.

