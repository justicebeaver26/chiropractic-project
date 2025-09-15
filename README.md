# chiropractic-project
This is my project on integrating integrating AI for predicting stress outcomes in chiropractic research.

# Integrating AI for Predicting Stress Outcomes in Chiropractic Research  

## 1. Purpose  
The purpose of this project is to evaluate the impact of chiropractic care on stress outcomes.  
Both physiological (cortisol levels) and psychological (DASS-21 scores) measures were analyzed in adults and children over 12 weeks.  
The study aimed to determine which factors best predict stress responses and whether chiropractic interventions influence them.  

## 2. Data Collection  
- Data was collected from a 12-week chiropractic intervention study including adults and children.  
- Measures included:  
  - **Salivary cortisol and Blood Cortisol (for adults) and Salivary cortisol and Hair cortisol (for children)** (physiological stress marker).  
  - **DASS-21** (psychological stress scale).  
  - **Subluxation scores** and **treatment groups**.  

## 3. Data Cleaning  
- Ensured all participants had valid baseline, intervention, and follow-up data.  
- Standardised variable formats (numeric for scores, categorical for groups).  
- Addressed missing values and verified dataset consistency.  

## 4. Modelling  
- **Linear Mixed-Effects Models** were used to capture repeated measures across time.  
- **Machine Learning Models** were applied for predictive analysis:  
  - Random Forest  
  - Gradient Boosting  
  - Support Vector Regression (SVR)  

## 5. Dashboards Showing Key Metrics  
- Comparative plots of cortisol and DASS scores before, during, and after intervention.  
- Visualizations of model performance and feature importance.  
- Separate dashboards for adults vs children to highlight differences.  

## 6. Conclusion  
- **Cortisol levels** increased post-intervention, while **DASS-21 scores** declined — suggesting different physiological and psychological responses.  
- **Baseline cortisol and DASS scores** were stronger predictors of stress outcomes than subluxation scores or treatment groups.  
- Results for **children’s data** were inconclusive, indicating the need for larger samples and longer-term studies.  
- Overall, findings suggest cortisol response may be more biologically driven and less impacted by short-term chiropractic care.
