package myml

import mymllib.MultiClassEvaluation
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object PerceptronPipeline {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("PerceptronPipeline").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname: cleveland or ann
    //    val dataname = "cleveland"
    val dataname = "acs_bayes"
    //    val dataname = "acs_43"
    //    val dataname = "ann"

    // Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "undersampling.libsvmForm")

    // Split the data into train and test
    var Array(train, test) = data.randomSplit(Array(0.8, 0.2))

    //oversampling
    new OverSampling(train).os
    train = sqlContext.read.format("libsvm").load("processedData/tmp.train.oversampling.libsvmForm")


    val datasetClass = data.select("label").collect().distinct
    val classnum = datasetClass.size
    val featureSize = data.first().getAs[org.apache.spark.mllib.linalg.Vector]("features").size

    // create trainer and pipeline
    val trainer = new MultilayerPerceptronClassifier()
      .setBlockSize(128)
      .setSeed(1234L)
    val pipeline = new Pipeline().setStages(Array(trainer));


    //cross validation. Specify layers for the neural network
    val kfold = 10
    val paramGrid = new ParamGridBuilder()
      //      .addGrid(trainer.layers, Array(Array(featureSize, 15, 8, classnum), Array(featureSize, 20, 14, 8, classnum)).toIterable) //for acs_bayes_ori
      //      .addGrid(trainer.layers, Array(Array(featureSize, 11, 6, classnum), Array(featureSize, 12, 8, 4, classnum)).toIterable) //for acs_bayes_16
      .addGrid(trainer.layers, Array(Array(featureSize, 18, 10, classnum), Array(featureSize, 20, 14, 8, classnum)).toIterable) //for acs_bayes_16
      //      .addGrid(trainer.layers, Array(Array(featureSize, 28, 15, classnum), Array(featureSize, 31, 21, 11, classnum)).toIterable) //for acs_bayes_16

      //      .addGrid(trainer.layers, Array(Array(featureSize, 50, 20, classnum), Array(featureSize, 55, 35, 15, classnum)).toIterable) //for cleveland
      //      .addGrid(trainer.layers, Array(Array(featureSize, 15,  9, classnum), Array(featureSize, 11, classnum)).toIterable) //for ann
      .addGrid(trainer.maxIter, Array(5)) //150
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(kfold)
    val cvModel = cv.fit(train)

    println("cvClass:" + cvModel.getClass)

    // Make predictions, which is based on majority vote for classfication problem.
    val predictions = cvModel.transform(test)
    println(predictions.getClass)
    predictions.show(5)
    predictions.printSchema()

    //evaluation
    //multiclass
    val mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
      (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
    new MultiClassEvaluation(mypredictionAndLabels, datasetClass)

    //print params for best model
    cvModel.bestModel.params.foreach(println)
    val paramMap = cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1
    println(paramMap)
    val MLPModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(0).
      asInstanceOf[MultilayerPerceptronClassificationModel]
    println("Layers for best model:")
    println(MLPModel.layers.toList)

    sparkContext.stop()
  }
}