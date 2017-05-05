package mymllib

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object ParquetToLabeledPoint {
  //output LabeledPoint is RDD[LabeledPoint] in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ParquetToLabeledPoint").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger().setLevel(Level.ERROR)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //    val dataname = "acs_16"
//    val dataname = "MACCE_64"
    //val dataname = "MACCE_d36"
    //val dataname = "MACCE_d14"
//    val dataname = "acs_39_combined" //26 28 29 31 33 34 36 37 39
//    val dataname = "acs_16_possible"
//    val dataname = "acs_13_possible"
//    val dataname = "acs_42_ori" //42 - 26
    //    val dataname = "acs_bayes_16"
    //    val dataname = "acs_whole"
//       val dataname = "acs_42"

        val dataname = "cleveland"
    //    val dataname = "ann"
    val parquetFile = sqlContext.read.parquet("processedData/" + dataname + ".parquet")
    println(parquetFile.getClass)

    //Assemble Feature Vector
    val featurename = parquetFile.schema.fieldNames.filter(fname => (fname != "label"))
    val assembler = new VectorAssembler()
      .setInputCols(featurename)
      .setOutputCol("features")
    val WithCombinedFeature = assembler.transform(parquetFile)

    //Ｃonstruct RDD[Labeledpoint], in the form of row(label, vector), for LR
    val labeleddata = WithCombinedFeature.map(row => LabeledPoint(row.getAs[Double]("label"),
      row.getAs[org.apache.spark.mllib.linalg.Vector]("features"))
    )
    println("----data convert:---")
    println(labeleddata.getClass)
    labeleddata.take(3).foreach(println)

    //save file, delete first if exists
    val libsvmFormPath = "processedData/" + dataname + ".libsvmForm"
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