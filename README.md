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
├─ notebooks/
│   └─ exploratory/     
│
├─ src/            
│
├─ models/        
│
├─ outputs/        
│
└─ README.md       
```
* data/raw/ orginal csv files from kaggle, already separated into a training ang test set
* data/processed/ cleaned and augmented files that will be used to train and test the models
* notebooks/exploratory/ notebook files used for data exploration
* src/ .py pipeline script that that will execute preprocessing, training, and evaluation
* models/ trained models such as logistic regression
* figures/ plots and other visualizations created for analysis

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