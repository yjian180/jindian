package myml

import mymllib.MultiClassEvaluation
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object RandomForestClasification {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClasification").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    // Load and parse the data file, converting it to a DataFrame, default column names are "label" and "features".
           val dataname = "cleveland"
    //    val dataname = "ann"
    //    val dataname = "acs_bayes"
    //    val dataname = "acs_16"
    //    val dataname = "acs_26"
    //val dataname = "acs_43"
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "undersampling.libsvmForm")
    // Split the data into training and test sets (30% held out for testing)
    var Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

    //oversampling
    new OverSampling(trainingData).os
    trainingData = sqlContext.read.format("libsvm").load("processedData/tmp.train.oversampling.libsvmForm")

    val datasetClass = data.select("label").collect().distinct
    val classnum = datasetClass.size

    // Index labels, adding metadata to the label column.Off
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(data)
    // Set RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    //cross validation, what need to be change is the numTrees param and k-fold.
    val kfold = 10
    val paramGrid = new ParamGridBuilder()
      .addGrid(featureIndexer.maxCategories, Array(classnum))
      //      .addGrid(rf.numTrees, Array(5, 10, 15, 20, 25, 30, 35, 40))
      //      .addGrid(rf.numTrees, Array(1, 2, 3, 4, 5, 10, 15, 20, 25))   //acs_26
      .addGrid(rf.numTrees, Array(1, 3, 5, 10, 20)) //acs_16
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(kfold)
    val cvModel = cv.fit(trainingData)

    //print forests for decision
    val rfModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + rfModel.toDebugString)


    // Make predictions, which is based on majority vote for classfication problem.
    val predictions = cvModel.transform(testData)


    //evaluation
    //multiclass
    val mypredictionAndLabels = predictions.select("prediction", "label").map(row =>
      (row.getAs[Double]("prediction"), row.getAs[Double]("label")))
    new MultiClassEvaluation(mypredictionAndLabels, datasetClass)
    //binary evaluation. Clear the prediction threshold so the model will return probabilities
    new BinaryClassEvaluation(predictions)

    //output parameters of the best model
    cvModel.bestModel.params.foreach(println)
    val paramMap = cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1
    println(paramMap)


    sparkContext.stop()
  }
}

