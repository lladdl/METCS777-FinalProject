# METCS777-FinalProject
# **Chicago Crime Arrest Prediction**

This project uses the Chicago Crime dataset to build a classification model that predicts whether a reported crime results in an arrest. The focus is on understanding key predictors, preprocessing a large dataset efficiently, and implementing machine-learning models for arrest prediction.

---

## ** Dataset**

**Source:** Chicago Crime Data (2001–Present)
Link: [https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000?select=Chicage+Crime+Data.csv](https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000?select=Chicage+Crime+Data.csv)

The dataset includes:

* Crime details: `Primary Type`, `Description`, `IUCR`
* Location: `Beat`, `District`, `Ward`, `Community Area`, `Latitude`, `Longitude`
* Time: `Date`, `Year`, `Month`, `Day`, `Hour`, `Season`
* Flags: `Arrest` (target variable), `Domestic`

**Target Variable:**
`Arrest` — binary outcome indicating whether an arrest was made.

**Size:**
~8 million rows, multi-GB file

---

## ** Coding Logic / Modeling Pipeline**

### **1. Load and Clean the Data**

* Read the large CSV using Spark or chunked pandas (depending on environment)
* Remove rows with invalid coordinates or missing essential fields
* Parse the `Date` column into:

  * `year`, `month`, `day`, `hour`, `weekday`, `season`
* Filter obvious data errors (e.g., out-of-range values)

### **2. Feature Engineering**

* Encode categorical variables (Primary Type, Beat, District, etc.)
* Create geospatial groupings (optional):

  * Community Area
  * Police Beat
* Convert temporal variables into cyclical encodings if needed
* Balance classes if arrests are rare (SMOTE or class weights)

### **3. Train/Test Split**

* Split into training and test sets
* Ensure stratification by `Arrest` to maintain class distribution

### **4. Modeling**

Three main models:

#### **Logistic Regression**

*Baseline Model

#### **Random Forest**

* Captures nonlinear interactions
* Provides interpretable feature importance

#### **XGBoost**

* Gradient-boosted trees for improved performance
* Handles imbalanced classes using scale_pos_weight
* Produces probabilities for ROC/AUC evaluation

### **5. Evaluation Metrics**

* Accuracy
* Precision / Recall
* F1 Score
* ROC AUC
* Error analysis:

  * False positives (model predicts arrest when none occurred)
  * False negatives (model misses true arrests)
 
### **6. Mapping**

 * Shape file of the community areas of Chicago used to map metrics
 * Shape file and metrics csvs are used to create the visualizations
 






