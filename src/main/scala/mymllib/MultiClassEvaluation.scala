package mymllib

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

/**
  * Created by yj on 17-1-20.
  */
class MultiClassEvaluation(mypredictionAndLabels: RDD[(Double, Double)], datasetClass: Array[Row]) {

  //multiclass evaluation metric

  //precision, fmeasure, recall, falsepositive
  val metrics = new MulticlassMetrics(mypredictionAndLabels)
  val precision = metrics.precision
  val fmeasure = metrics.fMeasure
  val recall = metrics.recall
  val classnum = datasetClass.size
  //accuracy
  val labelCountByClass = mypredictionAndLabels.values.countByValue()
  val labelCount = labelCountByClass.values.sum
  var singleClassEvaluation = Array.ofDim[Double](4, classnum)
  for (i <- 0 until classnum) singleClassEvaluation(0)(i) = metrics.precision(datasetClass(i).getDouble(0))
  for (i <- 0 until classnum) singleClassEvaluation(1)(i) = metrics.fMeasure(datasetClass(i).getDouble(0))
  for (i <- 0 until classnum) singleClassEvaluation(2)(i) = metrics.recall(datasetClass(i).getDouble(0))
  for (i <- 0 until classnum) singleClassEvaluation(3)(i) = metrics.falsePositiveRate(datasetClass(i).getDouble(0))

  //print evaluation
  println("Precision\tF1\trecall:")
  println(precision + "\t" + fmeasure + "\t" + recall)
  println("Evaluation for Single precision, fmeasure, recall, falsepositive")
  println("ordered by label: ")
  for (i <- 0 until classnum) {
    print(datasetClass(i).getDouble(0) + "\t")
  }
  println
  for (i <- 0 until singleClassEvaluation.length) {
    for (j <- 0 until classnum) {
      print(singleClassEvaluation(i)(j) + "\t")
    }
    println
  }
  val tpByClass = mypredictionAndLabels
    .map { case (prediction, label) =>
      (label, if (label == prediction) 1 else 0)
    }.reduceByKey(_ + _)
    .collectAsMap()
  val accuracy = tpByClass.values.sum.toDouble / labelCount


  println("accuracy:" + accuracy)

}
