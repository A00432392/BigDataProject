#import required libraries

from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

#split data to train model
sc = SparkContext()
spark = SQLContext(sc)
#Datapipeline for MLlib
pipeline_load = PipelineModel.load("/user/trained_pipeline/")
ddf=spark.read.csv("/__dsets/FinalNLP-2.txt",header=True,sep='\t')
predictions = pipeline_load.transform(ddf)
columns = ['type','time','text','probability','prediction']
count=predictions.count()
predictions[columns].toPandas().to_csv("/__dsets/predicted_nlp.csv")
