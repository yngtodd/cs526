"""To Be Continued..."""
import pyspark
import numpy as np
import argparse

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *

from skopt import gp_minimize


#initialize spark session
spark = SparkSession\
        .builder\
        .appName("Hyperparameter Optimization")\
        .config('spark.sql.warehouse.dir', 'file:///C:/')\
        .getOrCreate()

sc = spark.sparkContext

parser = argparse.ArgumentParser(description='Pyspark Training')
parser.add_argument('--data', type=str,
    default="../../../data/sample_linear_regression_data.txt",
    help='Data location.')
args = parser.parse_args()

# Grab the regression dataset
data = spark.read.format("libsvm").load(args.data)

# Create training, validation, and test splits
(train, test) = data.randomSplit([0.8, 0.2])
(train, val) = train.randomSplit([0.8, 0.2])


def objective(hparams):
    """
    Objective function to be minimized:
    Model validation RMSE loss as a function of our model hyperparameters.

    Parameters:
    ----------
    * `hparams` [list]
        Hyperparameter settings determined by Bayesian optimization loop.

    Returns:
    -------
    * `rmse` [float]
        Root mean squared error on the validation set    

    Reference:
    ---------
    Bayesian optimization with Scikit-Optimize:
    https://scikit-optimize.github.io/
    """
    # New hyperparameter settings from Bayesian optimization
    num_trees, max_depth, max_bins = hparams
    
    # Instantiate a RandomForest model.
    rf = RandomForestRegressor(numTrees=num_trees, maxDepth=max_depth, maxBins=max_bins)

    # Train model. 
    model = rf.fit(train)

    # Make predictions.
    predictions = model.transform(val)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    
    rmse = evaluator.evaluate(predictions)
    #print('Validation RMSE: {}'.format(rmse))
    return rmse


def main():
    # Set bounds for random forest's hyperparameters
    hparams = [(2, 25),   # num_trees
               (2, 6),    # max_depth
               (15, 30)]  # max_bins
    
    # Run hyperparameter optimization using Gaussian processes
    optim_results = gp_minimize(objective, hparams, n_calls=20, verbose=True, random_state=0)
    print('\nHyperparameter Optimization Results:')
    print('Best validation RMSE = {}'.format(optim_results.fun))

    # Get best hyperparameters from optimization
    num_trees = optim_results.x[0]
    max_depth = optim_results.x[1]
    max_bins = optim_results.x[2]
    
    # Instantiate a RandomForest model using best hyperparameter settings
    rf = RandomForestRegressor(numTrees=num_trees, maxDepth=max_depth, maxBins=max_bins)

    # Train model. 
    model = rf.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    rmse = evaluator.evaluate(predictions)
    print('\nFinal Results on Test Set with Optimized Hyperparameters:')
    print("Root Mean Squared Error on test set = %g" % rmse)


if __name__=="__main__":
    main()
