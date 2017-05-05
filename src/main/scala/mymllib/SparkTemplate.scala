package mymllib

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object SparkTemplate {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkTemplate").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    val a = Array(1, 2, 3)
    var b = Array(4, 5, 6)

    //    val c  = b.iterator
    //    val d : Iterable[Array[Int]] = b.toIterable
    println(b.getClass)
    //    println(d.getClass)


    val c = new Array[Array[Int]](5)
    for (i <- 0 until c.length) {
      c(i) = Array(i, 2, 4)
      println(c(i).getClass)
      c(i).foreach(println)
    }
    println(c.getClass)

    //    val d =  Iterable[Array[Int]]
    sparkContext.stop()
  }

}