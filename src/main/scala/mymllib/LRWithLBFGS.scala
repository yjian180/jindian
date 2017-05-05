package mymllib

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object LRWithLBFGS {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LRWithLBFGS").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //Load file as RDD[LabeledPoint]
    //val dataname = "acs_bayes"
    val dataname = "acs_39_combined"
    //    val dataname = "ann"
    for (i <- 0 until 100) {
      val split = Array(0.8, 0.2)
    //    val dataname = "ann"

    //data types includes: original, oversampling, and different size of Chisq or PCA features
    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + ".libsvmForm")
    //        val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + "oversampling.libsvmForm")
    //    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + "undersampling.libsvmForm")
    //difference feature size, which is 5, 10, 15 for ANN; 18, 37, 56 for Cleveland
    //    val newFeatureSize=18
    //    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + ".Chisq."+ newFeatureSize+"oversampling.libsvmForm")
    //    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + ".Chisq."+ newFeatureSize+".libsvmForm")
    //    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + ".PCA."+ newFeatureSize+".libsvmForm")
    //    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + ".PCA."+ newFeatureSize+"oversampling.libsvmForm")
    //

    val datasetClass = sqlContext.createDataFrame(data).select("label").collect().distinct
    val classnum = datasetClass.size

      // Split data into training (70%) and test (30%).
      val splits = data.randomSplit(split)
      var training = splits(0).cache()
      //    training = new OverSampling(training)

      //oversampling
      //    new OverSampling(training).os
      //    training = MLUtils.loadLibSVMFile(sparkContext, "processedData/tmp.train.oversampling.libsvmForm")

//      println("train class:" + training.getClass)
      val test = splits(1)

      // Run training algorithm to build the model
      val model = new LogisticRegressionWithLBFGS()
        .setNumClasses(classnum)
        .setValidateData(false)
        .run(training)

      // Compute raw scores on the test set.
      val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }

      // Save and load model, delete first if exists
      val modelPath = "processedData/" + dataname + ".LR.model"
      val modelFile = new File(modelPath)
      delete(modelFile)
      model.save(sparkContext, modelPath)
      val sameModel = LogisticRegressionModel.load(sparkContext, modelPath)

//      val ＣolumnＴypeＷeightＭatrix: Matrix = Matrices.dense(model.numFeatures, model.numClasses - 1, model.weights.toArray)
//      val weightＭatrix: Matrix = ＣolumnＴypeＷeightＭatrix.transpose
//      println("weight for each label:")
//      for (i <- 0 until model.numClasses - 1) {
//        for (j <- 0 until model.numFeatures) {
//          print(weightＭatrix(i, j) + "\t")
//        }
//        println()
//      }

      //evaluation
      //multiclass
//      new MultiClassEvaluation(predictionAndLabels, datasetClass)
      //binary evaluation. Clear the prediction threshold so the model will return probabilities
      model.clearThreshold
      val BiPredictionAndLabels = test.map { case LabeledPoint(label, features) =>
        val biprediction = model.predict(features)
        (biprediction, label)
      }
      new BinaryClassEvaluation(BiPredictionAndLabels)
    }

    sparkContext.stop()
  }

  private def delete(file: File) {
    if (file.isDirectory)
      Option(file.listFiles).map(_.toList).getOrElse(Nil).foreach(delete(_))
    file.delete
  }

}