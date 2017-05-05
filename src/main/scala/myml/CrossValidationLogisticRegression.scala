package myml

import mymllib.MultiClassEvaluation
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object CrossValidationLogisticRegression {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegression").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname
    val dataname = "acs_16"
    //        val dataname = "acs_26"
    //    val dataname = "acs_43"
    //        val newFeatureSize = 26   //5,11,16,21,27,32，38 forareaUnderROC:0.756987499999997 acs_43(1/8 -7/8) (16,26 for baseline comparision)
    //    val dataname = "acs_bayes_16"

    //Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //        val data = sqlContext.read.format("libsvm").load("processedData/" + dataname +  ".Chisq."+ newFeatureSize+".libsvmForm")

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


    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(0).asInstanceOf[LogisticRegressionModel]

    val ＣolumnＴypeＷeightＭatrix: Matrix = Matrices.dense(lrModel.numFeatures, lrModel.numClasses - 1, lrModel.coefficients.toArray)
    val weightＭatrix: Matrix = ＣolumnＴypeＷeightＭatrix.transpose
    println("weight for each label:")
    for (i <- 0 until lrModel.numClasses - 1) {
      for (j <- 0 until lrModel.numFeatures) {
        print(weightＭatrix(i, j) + "\t")
      }
      println()
    }


    //evaluation
    //multiclass
    val mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
      (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
    val datasetClass = data.select("label").collect().distinct
    new MultiClassEvaluation(mypredictionAndLabels, datasetClass)
    //binary evaluation. Clear the prediction threshold so the model will return probabilities
    new BinaryClassEvaluation(predictions)


    sparkContext.stop()
  }

}