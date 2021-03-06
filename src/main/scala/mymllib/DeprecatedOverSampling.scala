package mymllib

import java.io.File

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable


object DeprecatedOversampling {
  //output LabeledPoint is RDD[LabeledPoint] in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DeprecatedOversampling").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    val dataname = "acs_bayes"
    //    val dataname = "cleveland"
    //    val dataname = "ann"
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    println("data class:" + data.getClass)

    val rddData = data.map(row => (row.getAs[Double]("label"),
      row.getAs[org.apache.spark.mllib.linalg.Vector]("features"))
    ).cache()

    val keyCount = rddData.countByKey() //hashmap
    println("count for each class:")
    keyCount.foreach { case (key, value) => println("class=" + key + ", count=" + value) }
    val maxCount = keyCount.valuesIterator.max
    val minCount = keyCount.valuesIterator.min
    val time = maxCount / minCount + 1

    val fractions: collection.mutable.Map[Double, Double] = new mutable.HashMap[Double, Double]()
    keyCount.keys.foreach(key => {
      fractions(key) = ((maxCount.toDouble - keyCount.get(key).get) / time / keyCount.get(key).get)
    })

    var overRDD = rddData
    for (i <- 0 until time.toInt) {
      val newdata = rddData.sampleByKeyExact(true, fractions)
      overRDD = overRDD ++ newdata
    }
    val osKeyCount = overRDD.countByKey() //hashmap
    println("count after over sampling for each class:")
    osKeyCount.foreach { case (key, value) => println("class=" + key + ", count=" + value) }

    //if not set features to dense, the last features vector with 0 value will be discarded.
    val labeleddata = overRDD.map(rdd => LabeledPoint(rdd._1, rdd._2.toDense))

    //save file, delete first if exists
    val libsvmFormPath = "processedData/" + dataname + "oversampling.libsvmForm"
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