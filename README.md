## American Express Credit Card Default Prediction

# Description
This project will use American Express credit card data aqcuired from kaggle to build machine learning models to predict if an individual will default on their credit card. The goal is to eexplore the dataset, perform feature engineering, and apply ML techniques like logistic regression and classification to provide accurate predictions of credit risk. This project will demonstrate the workflow on an end-to-end project starting with exploration and concluding in analysis of results.

# Directory Structure
```
cmse492_project/
│
├─ data/
│   ├─ raw/        
│   └─ processed/ 
│
├─ firgures/
│
├─ notebooks/
│   └─ exploratory/     
│
├─ src/  
│   ├─ evaluation/
│   ├─ models/        
│   └─ preprocessing/ 
│
├─ trained_models/        
│
├─ requirements.txt   
│
└─ README.md       
```
* data/raw/ orginal dataset obtained from kaggle
* data/processed/ two files, both processed and imputed, one has synthetic entries to aid in training
* figures/ plots and other visualizations created for analysis
* notebooks/exploratory/ initial exploration of dataset and baseline logistic regression
* src/evaluation/ notebook testing and comparing all three models on dataset without synthetic entries
* src/models/ notebooks containing the training and evaluation of models on dataset with synthetic entries
* src/preprocessing/ two files, one that cleans and imputes dataset, another that uses SMOTE to add synthetic entries
* trained_models/ all 3 trained models are saved and located here as either a .pkl or .cbm

# Setup Instructions
1. Clone Repository
  ```
git clone https://github.com/TrevorBloom/cmse492_project.git
cd cmse492_project
  ```

2. Create Virtual Environment
```
# On Windows
python -m venv venv
# On macOS/Linux
venv\Scripts\activate

source venv/bin/activate
```

3. Install Dependencies
```
pip install -r requirements.tct
```

4. Run Scripts
```
Will be updated to correctly give instructions as project is worked on/completed
```
