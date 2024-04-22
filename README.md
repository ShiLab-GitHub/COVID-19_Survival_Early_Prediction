# Early Prediction of COVID-19 Patient Survival by  Blood Plasma using Machine Learning
The Coronavirus Disease 2019 (COVID-19) pandemic has severely disrupted the global healthcare and medical system. Although COVID-19 is no longer considered a public health emergency of international concern, it can still cause many infections and even life-threatening conditions. This repository contains the data and python script in support of manuscript: Early Prediction of COVID-19 Patient Survival by  Blood Plasma using Machine Learning. The focus of this study is to reveal novel potential biomarkers of mortality and identify associated mechanisms of death caused by COVID-19 using machine learning approaches to re-analyze metabolomics data. 

# Data
Data sets are stored in the data file.

The processed metabolomics data used in this study were obtained from Figshare (https://doi.org/10.6084/m9.figshare.22047761.v1). The dataset consisting of 197 samples from 177 individuals, including 28 samples from patients with a mild diagnosis of COVID-19, 40 samples from moderate COVID-19 patients, 49 samples from severe COVID-19 patients, 33 samples from patients who progressed to a fatal outcome, and 27 samples from control donors. 20 samples from a group of individuals with longitudinal follow-up who allowed paired sampling on average of 172 days after recovery from COVID-19.
Annotated information and detailed patient descriptions are available in the original paper's supplementary material [1].

[1] L. G. Gardinassi et al., “Integrated Metabolic and Inflammatory Signatures Associated with Severity of, Fatality of, and Recovery from COVID-19,” MICROBIOLOGY SPECTRUM, vol. 11, no. 2, Apr. 2023.

# Model
We built the model to predict the clinical outcome of COVID-19 based on the data set containing 82 samples (49 samples from severe patients diagnosed with COVID-19 and 33 samples from patients with progression to fatal outcome).

Our study fitted two prediction models, random forest (RF) and lightGBM (LGB), to the entire cohort through a 10-fold cross-validation framework, and evaluated the prediction performance by measuring the area under the curve (AUC). RF combines a set of decision trees into an "integrated" learner with multiple trees, whose output categories are determined by the plurality of output categories of each tree, resulting in stronger output predictions. LGB is a tree based Ensemble learning method that uses Gradient boosting to combine multiple weak learners into a powerful model.

To predict the likelihood of survival, we analyzed and identified important metabolites of COVID-19 patients. In this study, ANOVA, correlation analysis and SHAP analysis were used to explore the feature metabolites. Based on the RF and LGB models, the top 20 SHAP metabolites were screened and the intersected feature metabolites were identified as the most important metabolites in both algorithms for subsequent analysis.

# Dependencies
```
Python 3.7.16
Scikit 1.0.2
Numpy 1.21.5
Pandas 1.3.5
Shap	 0.41.0
Matplotlib 3.5.3
```

# Result
The pictures and results were saved in ./data/shap/01

# Citation
```
@inproceedings{Zhu, Yibo;Shi, Xiumin;Wang, Yan;Zhu, Yixuan;Wang, Lu2023Early Prediction of COVID-19 Patient Survival by Blood Plasma Using Machine Learning,
title={Early Prediction of COVID-19 Patient Survival by Blood Plasma Using Machine Learning},
author={Zhu, Yibo;Shi, Xiumin;Wang, Yan;Zhu, Yixuan;Wang, Lu},
booktitle={6th IEEE International Conference on Computer and Communication Engineering Technology, CCET 2023},
year={2023},
}
```

