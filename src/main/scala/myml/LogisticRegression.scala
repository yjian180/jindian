package myml
import mymllib.MultiClassEvaluation
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object LogisticRegression {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegression").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname
    //    val dataname = "acs_16"
    //    val dataname = "acs_26"
    val dataname = "MACCE_d36"
    //val dataname = "cleveland"
    //val dataname = "acs_43"
   val newFeatureSize = 32 //5,11,16,21,27,32，38 for acs_43(1/8 -7/8)
    //    val dataname = "acs_bayes_16"

    // Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
    //        val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".Chisq." + newFeatureSize + ".libsvmForm")

    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "undersampling.libsvmForm")

    var Array(train, test) = data.randomSplit(Array(0.8, 0.2))

    //oversampling
    //    new OverSampling(train).os
    //    train = sqlContext.read.format("libsvm").load("processedData/tmp.train.oversampling.libsvmForm")

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStandardization(true)
      .setRegParam(1.0)
      .setElasticNetParam(0)
      .setThreshold(0.11)

    val pipeline = new Pipeline()
      .setStages(Array(lr))
    val model = pipeline.fit(train)


    //    val model = lr.fit(train)


    val predictions = model.transform(test)
    predictions.printSchema()

    //evaluation
    //multiclass
    val mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
      (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
    val datasetClass = data.select("label").collect().distinct
    new MultiClassEvaluation(mypredictionAndLabels, datasetClass)
    //binary evaluation. Clear the prediction threshold so the model will return probabilities
    new BinaryClassEvaluation(predictions)


   // println("probability and label:")
   // predictions.select("probability", "label").foreach(println)


    sparkContext.stop()
  }

}