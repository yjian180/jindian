package myml

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame

/**
  * Created by yj on 16-11-24.
  */
class BinaryClassEvaluation(predictions: DataFrame) {
  //binary evaluation metric
  val bimetric = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("probability")
    .setMetricName("areaUnderROC")
  val areaUnderROC = bimetric.evaluate(predictions)
//  println("areaUnderROC:")
  println("areaUnderRoc"+areaUnderROC)

  val bimetricPR = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("probability") //probability rawPrediction
    .setMetricName("areaUnderPR")
  val areaUnderPR = bimetricPR.evaluate(predictions)
  println("areaUnderPR:"+ areaUnderPR )
//  println(areaUnderPR)
}
