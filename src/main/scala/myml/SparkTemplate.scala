package myml

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object SparkTemplate {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkTemplate").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)


    sparkContext.stop()
  }

}