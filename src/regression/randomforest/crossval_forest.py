"""Example of Cross Validation and Grid Search for Model Hyperparameters"""
import pyspark
import numpy as np
import argparse

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *


#initialize spark session
spark = SparkSession\
        .builder\
        .appName("Test")\
        .config('spark.sql.warehouse.dir', 'file:///C:/')\
        .getOrCreate()

sc = spark.sparkContext


def main():
    parser = argparse.ArgumentParser(description='Pyspark Training')
    parser.add_argument('--data', type=str, 
        default="../../../data/sample_linear_regression_data.txt",
        help='Data location.')
    parser.add_argument('--cross_val', type=bool, default=False, help='whether to use cross_validation')
    args = parser.parse_args()

    data = spark.read.format("libsvm").load(args.data)

    # Split the data into training and test sets (30% held out for testing)
    (train, test) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestRegressor()
    
    # Create a grid of hyperparameters. Each combination will be tested.
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.numTrees, [2, 25]) \
        .addGrid(rf.maxDepth, [2, 6])\
        .addGrid(rf.maxBins, [15, 30])\
        .build()

    if args.cross_val:
        # Run five-fold cross validation to find best hyperparamters.
        crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
                          numFolds=5)  # use 3+ folds in practice

        model = crossval.fit(train)
    else:
        # Grid search for best hyperparameters with a single validation set.
        tvs = TrainValidationSplit(estimator=rf,
                                   estimatorParamMaps=paramGrid,
                                   evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
                                   # 80% of the data will be used for training, 20% for validation.
                                   trainRatio=0.8)
    
        # Run TrainValidationSplit, and choose the best set of parameters.
        model = tvs.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


if __name__=="__main__":
    main()
