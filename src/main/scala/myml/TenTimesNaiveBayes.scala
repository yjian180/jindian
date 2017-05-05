package myml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object TenTimesNaiveBayes {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NaiveBayes").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //specify dataname
    //        val dataname = "acs_26"
    //    val dataname = "acs_16"
    val dataname = "acs_42"
//    val dataname = "acs_39_combined"
    //val dataname = "MACCE_d36"
    //val dataname = "MACCE_d14"
   // val dataname = "cleveland"
    //  val dataname = "acs_whole"
    for (i <- 0 until 100) {
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
      val model = pipeline.fit(train)


      val predictions = model.transform(test)
      //      predictions.printSchema()

      //      println("all probability label")
      //      predictions.select("probability", "label").foreach(println)

      //    val nbt = new NaiveBayes()
      //      .setFeaturesCol("features")
      //      .setLabelCol("label")
      //      .setModelType("multinomial")
      //      .fit(train)
      //    nbt.transform(test)

      //evaluation
      //multiclass
      val mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
        (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
      val datasetClass = data.select("label").collect().distinct


      //      new MultiClassEvaluation(mypredictionAndLabels, datasetClass)
      //binary evaluation. Clear the prediction threshold so the model will return probabilities
      new BinaryClassEvaluation(predictions)



//      val predictionAndLabels = predictions.select("probability", "label").map(row =>
//        (row.getAs[Double]("probability"), row.getAs[Double]("label")))
//      val binaryMetrics = new BinaryClassificationMetrics(predictionAndLabels)
//      val auc = binaryMetrics.areaUnderROC
//      println(auc)

      //      println("------------Round "+i+" end------------------")
    }
    sparkContext.stop()
  }

}