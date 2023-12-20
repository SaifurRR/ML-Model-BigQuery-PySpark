# Initialize Spark
import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Read Parquet file
df = spark.read.parquet('shake.parquet') 

# Sample 10% of the data
df = df.sample(0.10)
df.show()

# Create a temporary view
df.createOrReplaceTempView("df")

# Import necessary libraries
from pyspark.sql.functions import monotonically_increasing_id
from systemds.context import SystemDSContext
import numpy as np
import pandas as pd

# Define function for calculating DFT using SystemDS
def dft_systemds(signal, name):
    with SystemDSContext(8080) as sds:
        size = len(signal)
        signal = sds.from_numpy(signal.to_numpy())
        pi = sds.scalar(3.141592654)

        n = sds.seq(0, size - 1)
        k = sds.seq(0, size - 1)

        M = (n @ (k.t())) * (2 * pi / size)
        
        Xa = M.cos() @ signal
        Xb = M.sin() @ signal

        index = (list(map(lambda x: [x], np.array(range(0, size, 1)))))
        DFT = np.hstack((index, Xa.cbind(Xb).compute()))
        DFT_df = spark.createDataFrame(DFT.tolist(), ["id", name+'_sin', name+'_cos'])
        return DFT_df

# Select data for class 0
x0 = spark.sql("SELECT X from df WHERE class = 0")
y0 = spark.sql("SELECT Y from df WHERE class = 0")
z0 = spark.sql("SELECT Z from df WHERE class = 0")

# Select data for class 1
x1 = spark.sql("SELECT X from df WHERE class = 1")
y1 = spark.sql("SELECT Y from df WHERE class = 1")
z1 = spark.sql("SELECT Z from df WHERE class = 1")

# Apply DFT to each class and axis
df_class_0 = dft_systemds(x0, 'x') \
    .join(dft_systemds(y0, 'y'), on=['id'], how='inner') \
    .join(dft_systemds(z0, 'z'), on=['id'], how='inner') \
    .withColumn('class', lit(0))
    
df_class_1 = dft_systemds(x1, 'x') \
    .join(dft_systemds(y1, 'y'), on=['id'], how='inner') \
    .join(dft_systemds(z1, 'z'), on=['id'], how='inner') \
    .withColumn('class', lit(1))

# Union the DataFrames
df_dft = df_class_0.union(df_class_1)

# Display the DataFrame
df_dft.show()

# Import necessary libraries for pipeline
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["xa", 'xb', 'ya', 'yb', 'za', 'zb'], outputCol='features')

from pyspark.ml.classification import GBTClassifier
classifier = GBTClassifier(labelCol="class", featuresCol="features", maxDepth=5, maxIter=10, seed=42)

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])

# Fit the pipeline to the data
model = pipeline.fit(df_dft)

# Make predictions
prediction = model.transform(df_dft)

# Display predictions
prediction.show()

# Evaluate the accuracy of the predictions
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("class")
binEval.evaluate(prediction) 

# Repartition the DataFrame and write predictions to a JSON file
prediction = prediction.repartition(1)
prediction.write.json('a2_m4.json')

