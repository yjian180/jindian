package feature

import org.apache.spark.ml.feature.ChiSqSelectorModel
object ChiSqSelectedFeatureIndex {

  def getSelectedFeatureIndex(ChiSqModel: ChiSqSelectorModel): Array[Int] = {
    val index = ChiSqModel.selectedFeatures
    println("Selected Features： " + index.mkString("\t"))
    //    index.foreach(println)
    index
  }
}