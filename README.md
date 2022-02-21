# data-scientist-assesment

## Folder Structure

### [data](data)
Contains data used in analysis. This folder also includes:

1. [Kenya Deliveries Cleaner.csv](data/Kenya&#Deliveries&#Cleaner.csv): Cleaned data without comma typos.
2. [merged_data.csv](data/merged_data.csv): Contains dataset created after merging.
3. [](data/formatted_data.csv): Contains data created after cleaning.
### [notebooks](notebooks):
1. [data_merging.ipynb](notebooks/data_merging.ipynb): Constains notebook that merges multiple data sources.
2. [data_cleaning.ipynb](notebooks/data_cleaning.ipynb): Contains notebook that cleans merged data.
3. [data_analysis.ipynb](notebooks/data_analysis.ipynb): Contains notebook that analyzes the cleaned data and builds ML models.

### [scripts](scripts): Contains corresponding scripts for notebooks 
1. [data_merging.py](scripts/data_merging.py/): Contains module for merging data sources.
2. [data_cleaning.py](notebooks/data_cleaning.ipynb): Contains module for cleaning data.
3. [customer_retention.py](scripts/customer_retention.py): Contains module for customer retention analysis and prediction.
4. [customer_classification.py](scripts/customer_classification.py): Contains module for customer classification analysis and prediction.
5. [product_recommendation.py](scripts/product_recommendation.py): Contains module for customer retention analysis and prediction.
6. [revenue_optimization.py](scripts/revenue_optimization.py): Contains module for revenue optimization analysis and prediction.

### [requirements.txt](requirements.txt)
 Required libraries Python libraries

### [Report.pdf](Report.pdf)
 The report explaining my solution from data merging to building machine learning models

## Installation
1. Clone the repository.
2. In a virtual environment or global environment install the required libraries, run:
   `pip install -r requirements.txt `
3. To view my solutions
   1. Please view from the notebooks instead of the scripts. This more preferable to get visualizations and better explanations.
      1. Run - `jupyter-notebook `
      2. Run the notebooks
   2. Incase you want to run the scripts:
      1. Run (It is important you run within this directory because of file path errors)
         `cd scripts `
      2. Run:
        `python the_script_you_want_to_run `

## NB:
Run in the following order:
 1. Data merging
 2. Data Cleaning
 3. Data Analysis (Contains the ML models)

     