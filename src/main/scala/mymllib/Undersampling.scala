package mymllib

import java.io.File

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable


object Undersampling {
  //output LabeledPoint is RDD[LabeledPoint] in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Undersampling").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //    val dataname = "acs_bayes"
    val dataname = "acs_42"
    //    val dataname = "cleveland"
    //    val dataname = "ann"
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")


    val rddData = data.map(row => (row.getAs[Double]("label"),
      row.getAs[org.apache.spark.mllib.linalg.Vector]("features"))
    ).cache()

    val keyCount = rddData.countByKey() //hashmap
    println("count for each class:")
    keyCount.foreach { case (key, value) => println("class=" + key + ", count=" + value) }
    //    val maxCount = keyCount.valuesIterator.max
    val minCount = keyCount.valuesIterator.min
    //    val minCount = keyCount.valuesIterator.min*0.7
    //    val minCount = 600
    //    val time = maxCount / minCount + 1

    val fractions: collection.mutable.Map[Double, Double] = new mutable.HashMap[Double, Double]()
    keyCount.keys.foreach(key => {
      fractions(key) = (minCount.toDouble / keyCount.get(key).get)
    })

    //    var overRDD = rddData
    //    for (i <- 0 until time.toInt) {
    val UnderSamplingData = rddData.sampleByKeyExact(true, fractions)
    val usKeyCount = UnderSamplingData.countByKey() //hashmap
    println("count after under sampling for each class:")
    usKeyCount.foreach { case (key, value) => println("class=" + key + ", count=" + value) }

    //      overRDD = overRDD ++ newdata
    //    }
    //if not set features to dense, the last features vector with 0 value will be discarded.
    val labeleddata = UnderSamplingData.map(rdd => LabeledPoint(rdd._1, rdd._2.toDense))

    //save file, delete first if exists
    val libsvmFormPath = "processedData/" + dataname + "undersampling.libsvmForm"
    //    val libsvmFormPath = "processedData/" + dataname + "600.undersampling.libsvmForm"
    val libsvmFormFile = new File(libsvmFormPath)
    delete(libsvmFormFile)
    MLUtils.saveAsLibSVMFile(labeleddata, libsvmFormPath)

    sparkContext.stop()
  }

  private def delete(file: File) {
    if (file.isDirectory)
      Option(file.listFiles).map(_.toList).getOrElse(Nil).foreach(delete(_))
    file.delete
  }
}