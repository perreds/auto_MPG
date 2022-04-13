# Auto_MPG

This is the implementation and deployment of machine learning project predicts the miles per gallon of a car builded in the 80's and 90' atributes as cylinders, horsepower, weight, etc. Using the data set: http://archive.ics.uci.edu/ml/datasets/Auto+MPG

## Content

auto-mpg.data: Data set.

auto_MPG_EDA.ipynb: jupyter notebook with the exploratory data analysis.

auto_MPG_ML.ipynb: jupyter notebook with the preprocessing of the data and implementation of different ML models with hyperparameter optimization.

model_mpg.bin: final model

dir autoMPG_flask: folder with the scripts for deployment.

## Deployment using heroku

in ml_flask_app and the proper environment (in this case venv) create a git repository if is not already done, then log in in heroku account with

```bash
heroku login
```
then create the application

```bash
heroku create [name of application for url]
```
then push the repository to heroku

finally it will give the url to use in the request at the end of the auto_MPG_ML.ipynb file.



