Enterprise Data Science_COVID19
==============================

Enterprise Data Science 2020_COVID19
This project uses the data from John Hopkins data repo to visualise the covid spread on a dashboard developed using dash and plotly. It also provides an interactive environment to adjust the SIR curve for the COVID spread and predict the infection an recovery rates.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── notebooks          <- Jupyter notebooks. 
    |   |-- Business Understanding<- Goals and Constrainst of the project
    |   |-- Data Understanding <- John Hopkins data repository
    |   |-- Data Preparation   <- Relational data, Filter data
    |   |-- Modelling spread   <- Linear regression for doubling rate, 
    |   |-- Modelling Forecaset<- Simple forecasting using rolling mean from fb prophet
    |   |-- SIR Modelling      <- code for calculating SIR curves
    |   |-- Evaluation Walkthrough<- Displays dashboard integrating all the notebooks for         |   |                            covid spread
    |   |-- Evaluation Walkthrough SIR<-Displays dashboard integrating all the notebooks for       |   |                               SIR curve

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
