package myml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object TenTimesLogisticRegression {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegression").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger().setLevel(Level.ERROR)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname
    //    val dataname = "acs_16"
    //        val dataname = "acs_26"

    //    val dataname = "acs_26_combined"
    //    val dataname = "acs_39_combined" //26 28 29 31 33 34 36 37 39
    //    val dataname = "acs_42_ori" //42 - 26


//    val dataname = "acs_39_combined" //5	10	15	20	24	26	29	34	39
//    val dataname ="MACCE_d36"
    val dataname = "MACCE_d36"
    //        val dataname = "acs_42_ori"  //5,10,16,21,26,31,37, 42
    //              val newFeatureSize = 31//5,10,16,21,26,31,37 for acs_42(1/8 -7/8)
    //    val dataname = "acs_bayes_16"

    // Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
//    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //MACCE 73: 9 18 27 37 46 55 64
    //MACE_d14: 2 4 5 7 9 11 12
    //MACE_d36: 5 9 14 18 23 27 32

    val newFeatureSize = 32
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname +  ".Chisq."+ newFeatureSize+".libsvmForm")
//    //          val data = sqlContext.read.format("libsvm").load("processedData/" + dataname +  ".Chisq."+ newFeatureSize+".libsvmForm")

    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "undersampling.libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "600.undersampling.libsvmForm")
    for (i <- 0 until 100) {
      var Array(train, test) = data.randomSplit(Array(0.8, 0.2))

      //oversampling
      //          new OverSampling(train).os
      //          train = sqlContext.read.format("libsvm").load("processedData/tmp.train.oversampling.libsvmForm")

      var lr = new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setStandardization(true)
        .setRegParam(1.0)
        .setElasticNetParam(0)
        .setThreshold(0.9)

      var pipeline = new Pipeline()
        .setStages(Array(lr))
      var model = pipeline.fit(train)


      var predictions = model.transform(test)

      var lrModel = model.stages(0).asInstanceOf[LogisticRegressionModel]

      var ＣolumnＴypeＷeightＭatrix: Matrix = Matrices.dense(lrModel.numFeatures, lrModel.numClasses - 1, lrModel.coefficients.toArray)
      var weightＭatrix: Matrix = ＣolumnＴypeＷeightＭatrix.transpose
      //      println("weight for each label:")
      //      for (i <- 0 until lrModel.numClasses - 1) {
      //        for (j <- 0 until lrModel.numFeatures) {
      //          print(weightＭatrix(i, j) + "\t")
      //        }
      //        println()
      //      }

      //evaluation
      //multiclass
      var mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
        (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
      var datasetClass = data.select("label").collect().distinct
      //            new MultiClassEvaluation(mypredictionAndLabels, datasetClass)
      //binary evaluation. Clear the prediction threshold so the model will return probabilities
      new BinaryClassEvaluation(predictions)


      //          println("probability and label:")
      //          predictions.select("probability","label").foreach(println)

      //          println("------------Round "+i+" end------------------")

    }

    sparkContext.stop()
  }

}