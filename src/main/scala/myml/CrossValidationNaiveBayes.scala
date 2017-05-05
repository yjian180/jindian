package myml

import mymllib.MultiClassEvaluation
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object CrossValidationNaiveBayes {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("CrossValidationNaiveBayes").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname
    val dataname = "acs_bayes"

    // Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "undersampling.libsvmForm")

    var Array(train, test) = data.randomSplit(Array(0.8, 0.2))

    //oversampling
    //    new OverSampling(train).os
    //    train = sqlContext.read.format("libsvm").load("processedData/tmp.train.oversampling.libsvmForm")


    val nb = new NaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setModelType("multinomial")
    val pipeline = new Pipeline()
      .setStages(Array(nb))
    //    val model = pipeline.fit(train)


    //cross validation, what need to be change is the numTrees param and k-fold.
    val kfold = 10
    val paramGrid = new ParamGridBuilder()
      //      .addGrid(nb.smoothing,Array(1.0,2.0))
      .build()


    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(kfold)
    val cvModel = cv.fit(train)

    val predictions = cvModel.transform(test)
    predictions.printSchema()

    //evaluation
    //multiclass
    val mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
      (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
    val datasetClass = data.select("label").collect().distinct
    new MultiClassEvaluation(mypredictionAndLabels, datasetClass)
    //binary evaluation. Clear the prediction threshold so the model will return probabilities
    new BinaryClassEvaluation(predictions)
    cvModel.bestModel.params.foreach(println)
    sparkContext.stop()
  }

}