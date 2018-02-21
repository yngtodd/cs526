import numpy as np
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#initialize spark session
spark = SparkSession\
        .builder\
        .appName("Test")\
        .config('spark.sql.warehouse.dir', 'file:///C:/')\
       .getOrCreate()

sc = spark.sparkContext


def main():
    parser = argparse.ArgumentParser(description='Pyspark Training')
    parser.add_argument('--data', type=str, default="../../data/sample_libsvm_data.txt", 
                        help='Data location.')
    args = parser.parse_args()

    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load(args.data)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", 
                      outputCol="indexedFeatures", 
                      maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)
    print("Test Accuracy = {}".format(accuracy))
    print("Test Error = {}".format(1.0 - accuracy))

    gbtModel = model.stages[2]
    print(gbtModel)  # summary only


if __name__=="__main__":
    main()
