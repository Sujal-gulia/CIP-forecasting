#CPI Forecasting using Statistical and Deep Learning Models

This project aims to forecast the Consumer Price Index (CPI) of India using a combination of statistical and deep learning models. The motivation behind this work is to build a robust time series forecasting pipeline that incorporates macroeconomic indicators as predictors and offers insights into the future trends of inflation, which is crucial for economic planning and financial risk management.

The dataset used in this project is curated from public RBI sources and includes monthly records from October 2017 to June 2025. It spans a variety of economic features such as CPI (overall, food, rural, agricultural), Index of Industrial Production, foreign trade metrics, market borrowings, and exchange rate movements.

Dataset used can be acessed as follows: 

    from datasets import load_dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("Sujal08/cip-forecasting-Macroeconomic_Indicators_dataset")

The modeling pipeline starts with a univariate benchmark using AutoETS, which achieved a MAPE of 0.66%. To capture the influence of multiple economic variables, a multivariate VAR model was developed and tuned, reaching a MAPE of 1.54%. Finally, a deep learning approach using LSTM was implemented, yielding promising results with a MAPE of 0.89%. While the univariate model showed high performance, multivariate and LSTM models added interpretability and real-world alignment with economic signals.

| Model        | Type         | MAPE (%) |
|--------------|--------------|----------|
| AutoETS      | Univariate   | **0.66** |
| VAR          | Multivariate | **1.54** |
| LSTM         | Deep Learning | **0.89** |

This repository currently contains three core notebooks:
- `0_data_preproccesion.ipynb`: Prepares and cleans the data
- `1_eda&feature_engineering.ipynb`: Explores and engineers features
- `2_forecasting.ipynb`: Contains modeling logic and performance analysis
