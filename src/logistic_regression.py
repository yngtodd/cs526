import pyspark
import numpy as np
from splearn.rdd import ArrayRDD
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import numpy as np


def load_mnist(data_dir):
    """
    Load the MNIST dataset

    Parameters:
    ----------
    * `data_dir` [str]
        Location for the data.
        - If it does not exit, the data will be downloaded there.

    Returns:
    -------
    * `X` [nd-array shape=(70000, 784)]
        Handwritten digits data.
    * `y` [nd-array shape(70000,)]
        Labels.
    """
    mnist = fetch_mldata('MNIST original', data_home=data_dir)
    X = mnist['data']
    y = mnist['target']
    return X, y


def create_df(y, X):
    """
    Create Pyspark dataframe from numpy arrays.

    Parameters:
    ----------
    * `y` [numpy nd_array]
        Labels for the dataset.

    * `X` [numpy nd_array]
        Features for the dataset.

    Returns:
    -------
    Pyspark dataframe.
        - schema=["label", "features"]
    """
    stack = np.column_stack((y, X))
    data = map(lambda x: (int(x[0]), Vectors.dense(x[1:])), stack)
    return spark.createDataFrame(data, schema=["label", "features"])


def main():
    #initialize spark session
    spark = SparkSession\
            .builder\
            .appName("Test")\
            .config('spark.sql.warehouse.dir', 'file:///C:/')\
            .getOrCreate()

    sc = spark.sparkContext
    
    # Get the MNIST data.
    X, y = load_mnist('../data')

    # Create a train and test set split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
    
    # Convert numpy arrays to Pyspark dataframe.
    df = create_df(y_train, X_train)

    # Create a train and validation set.
    (train, val) = df.randomSplit([0.1, 0.90])
    
    # instantiate logistic regression with hyperparameters.
    lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

    # instantiate the One Vs Rest Classifier.
    ovr = OneVsRest(classifier=lr)
    
    # train the multiclass model
    ovrModel = ovr.fit(train)

    # score the model on test data.
    predictions = ovrModel.transform(val)

    # obtain evaluator.
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    # compute the classification error on validation data.
    accuracy = evaluator.evaluate(predictions)
    print("Validation accuracy = {}".format(accuracy))
    print("Test Error = %g" % (1.0 - accuracy))


if __name__=="__main__":
    main()
