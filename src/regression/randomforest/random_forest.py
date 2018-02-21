"""To Be Continued..."""
import pyspark
import numpy as np
import argparse

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

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
    args = parser.parse_args()

    data = spark.read.format("libsvm").load(args.data)

    # Split the data into training and test sets (30% held out for testing)
    (train, test) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestRegressor()

    # Train model.  This also runs the indexer.
    model = rf.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


if __name__=="__main__":
    main()
