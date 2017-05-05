package mymllib

import java.io.File

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object BinaryClassLRWithLBFGS {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BinaryClassLRWithLBFGS").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //Load file as RDD[LabeledPoint]
    val dataname = "acs_bayes"
    //    val dataname = "ann"
    val split = Array(0.7, 0.3)
    //    val dataname = "ann"

    //data types includes: original, oversampling, and different size of Chisq or PCA features
    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + ".libsvmForm")
    //    val data = MLUtils.loadLibSVMFile(sparkContext, "processedData/" + dataname + "oversampling.libsvmForm")
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
    val splits = data.randomSplit(split, seed = 11L)
    val training = splits(0).cache()
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

    // Get model summary and evaluation metrics.
    println("Summary = " + model.toString())
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    val fmeasure = metrics.fMeasure
    val recall = metrics.recall
    var singleClassEvaluation = Array.ofDim[Double](4, classnum)
    for (i <- 0 until datasetClass.size) singleClassEvaluation(0)(i) = metrics.precision(datasetClass(i).getDouble(0))
    for (i <- 0 until datasetClass.size) singleClassEvaluation(1)(i) = metrics.fMeasure(datasetClass(i).getDouble(0))
    for (i <- 0 until datasetClass.size) singleClassEvaluation(2)(i) = metrics.recall(datasetClass(i).getDouble(0))
    for (i <- 0 until datasetClass.size) singleClassEvaluation(3)(i) = metrics.falsePositiveRate(datasetClass(i).getDouble(0))


    // Save and load model, delete first if exists
    val modelPath = "processedData/" + dataname + ".LR.model"
    val modelFile = new File(modelPath)
    delete(modelFile)
    model.save(sparkContext, modelPath)
    val sameModel = LogisticRegressionModel.load(sparkContext, modelPath)

    //different form of model weights: vector, sparse/dense vector, weight matrix
    //    println("Weights = " + model.weights)
    //    println("Sparse Weights : = " + model.weights.toSparse)
    //    println("Dense Weights : = " + model.weights.toDense)

    val ＣolumnＴypeＷeightＭatrix: Matrix = Matrices.dense(model.numFeatures, model.numClasses - 1, model.weights.toArray)
    val weightＭatrix: Matrix = ＣolumnＴypeＷeightＭatrix.transpose
    println("weight for each label:")
    for (i <- 0 until model.numClasses - 1) {
      for (j <- 0 until model.numFeatures) {
        print(weightＭatrix(i, j) + "\t")
      }
      println()
    }

    // Get model summary and evaluation metrics.
    println("Summary = " + model.toString())
    //print evaluation
    println("Precision\tF1\trecall:")
    println(precision + "\t" + fmeasure + "\t" + recall)
    println("Evaluation for Single precision, fmeasure, recall, falsepositive")
    println("ordered by label: ")
    for (i <- 0 until datasetClass.size) {
      print(datasetClass(i).getDouble(0) + "\t")
    }
    println
    for (i <- 0 until singleClassEvaluation.length) {
      for (j <- 0 until classnum) {
        print(singleClassEvaluation(i)(j) + "\t")
      }
      println
    }

    //binary predict
    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold
    // Compute raw scores on the test set
    val BiPredictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val biprediction = model.predict(features)
      (biprediction, label)
    }

    val binaryMetrics = new BinaryClassificationMetrics(BiPredictionAndLabels)
    val auc = binaryMetrics.areaUnderROC
    val aupr = binaryMetrics.areaUnderPR
    println("Area Under ROC:"+auc)
    println("Area Under PR:"+aupr)


    sparkContext.stop()
  }

  private def delete(file: File) {
    if (file.isDirectory)
      Option(file.listFiles).map(_.toList).getOrElse(Nil).foreach(delete(_))
    file.delete
  }

}