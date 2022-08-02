# Development and validation of echocardiography-based machine-learning mortality prediction models and their correlation with patient-reported functional status

Notebooks and modules for training and testing CatBoost and deep neural networks for Echocardiography mortality analysis.
## Abstract

### Background: 
Echocardiography (echo) based machine learning (ML) models may be useful in identifying patients at high-risk of all-cause mortality.

### Methods: 
We developed ML models (ResNet deep learning using echo videos and CatBoost gradient boosting using echo measurements) to predict 1-year, 3-year, and 5-year mortality. Models were trained on the Mackay dataset, Taiwan (6083 echos, 3626 patients) and validated in the Alberta HEART dataset, Canada (997 echos, 595 patients). We examined the performance of the models overall, and in subgroups (healthy controls, at risk of heart failure (HF), HF with reduced ejection fraction (HFrEF) and HF with preserved ejection fraction (HFpEF)).  We compared models’ performance to the MAGGIC risk score, and examined the correlation between the models’ predicted probability of death and baseline quality of life measured by the Kansas City Cardiomyopathy Questionnaire (KCCQ).

### Findings: 
Mortality rates at 1-, 3- and 5-years were 14.9%, 28.6% and 42.5% in the Mackay cohort, and 3.0%, 10.3%, and 18.7%, in the Alberta HEART cohort. The ResNet and CatBoost achieved area under the receiver-operating curve (AUROC) between 85%-92% in internal validation. In external validation, the AUROCs for the ResNet (82%, 82%, and 78%) were significantly better than CatBoost (78%, 73%, and 75%), for 1-, 3- and 5-year mortality prediction respectively, with better or comparable performance to MAGGIC. ResNet models predicted higher probability of death in HFpEF and HFrEF (30% to 50%) subgroups than in controls and at risk patients (5% to 20%). The predicted probabilities of death correlated with KCCQ scores (all p<0.05). 

### Interpretation: 
ML models to predict mortality using echo data had good internal and external validity, were generalizable, correlated with patients’ quality of life, and performed better than or comparable to an established HF risk score. These models can be leveraged for echo-based automated risk stratification at an individual and population level.

# File Structure

```
├── CatBoost Models 
│   ├── Mackay_Mortality_1095days_CatBoost.ipynb
│   ├── Mackay_Mortality_1825days_CatBoost.ipynb
│   └── Mackay_Mortality_365days_CatBoost.ipynb
```

The CatBoost folder consists of the notebooks used to get the results of the CatBoost model along with the generation of the respective SHAP figures for 1 year,3 year and 5 years.
```
└── ResNet Models
    ├── Mackay_Mortality_DL_1095_days.ipynb
    ├── Mackay_Mortality_DL_1825_days.ipynb
    └── Mackay_Mortality_DL_365_days.ipynb
```

The Mackay and Alberta Data was pre-processed to remove unnecessary info and the notebooks for that are presnet in the Data Processing folder.

The ResNet folder consists of the notebooks used to get the results of the ResNet  model along with the generation of the respective  Activation Map figures for 1 year,3 year and 5 years.

# Requirements
The models were implemented using PyTorch 1.8.1  and CatBoost 0.26.1 in Python 3.8.6 with nvidia 2080Ti consisting of 11GB GDDR6.The requirements.txt file has further information of the various packages used.

# Getting Started
Initally the data might need to be pre-processed to remove unwanted data so the Data Processing files inside the Data Processing folder need to run to pre process the data to get only those patient ID that have PLAX view and remove unwanted columns that have very less data values for both Mackay and Alberta Data and to get the respective data for 1-year,3-year and 5-year using the Alberta_Mortality_Data_Processing.ipynb and Mackay_Mortality_Data_Processing.ipynb.

**input**: csv containing raw data of Alberta and the Mackay dataset<br />
**output**: cleaned csv files containing only patient ID with PLAX view and info regarding the mortality for 1-year,5-year and 3-year

PreProcessing.ipynb contains the pre-processing script for the video and to get the 16 frames for the ResNet model which could be used initially to get all the required frames from the videos.

**input** : csv file containing info about the patient ID<br />
**output**: frames of the PLAX view are created and saved in the respective patient ID folder.

After getting all the respective frames for each patient ID and the csv files , the CatBoost models could be run using the Mackay_Mortality_365days_CatBoost.ipynb ,Mackay_Mortality_1095days_CatBoost.ipynb and Mackay_Mortality_1825days_CatBoost.ipynb and each of them are customised seperately for the different timelines (1-year,3-year and 5-year). 

**input** : csv file containing info about the patients till 1 year ,3 year and 5 year<br />
**output**: model trained and weights saved along with shap figures.


Similary for the ResNet models , the following notebooks , Mackay_Mortality_DL_365_days.ipynb,Mackay_Mortality_DL_1095_days.ipynb,Mackay_Mortality_DL_1825_days.ipynb can be executed to get the model parameters for the different timelines (1-year,3-year and 5-year). 

**input** : csv file containing info about the patients till 1 year ,3 year and 5 year<br />
**output**: model trained and weights saved along with grad cam visualisations.

The various figures used in the paper along with the code used to generate them is present in the figures folder.Bootstrap.ipynb was used to generate figures related to the data bootstraping information for the model and the AUROC , PRC and other figures were generated from the figures notebook.
