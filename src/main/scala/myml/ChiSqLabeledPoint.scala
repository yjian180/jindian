package myml

import java.io.File
import feature.ChiSqSelectedFeatureIndex
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object ChiSqLabeledPoint {
  //output LabeledPoint is RDD[LabeledPoint] in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ChiSqLabeledPoint").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname: cleveland or ann
        val dataname = "cleveland"
//    val dataname = "acs_13_possible"
    //    val dataname = "ann"
//    val dataname = "acs_16_possible"
//    val dataname = "acs_39_combined"  //5	10	15	20	24	26	29	34	39
//        val dataname = "acs_42_ori"  //5,10,16,21,26,31,37, 42
//    val dataname = "MACCE2"
//    val dataname = "MACCE_d36"
    // Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")

    //Feature size should be less than 21 and 75 for ann and cleveland
    //We set it as 3/4, 1/2, 1/4 o5f original size, which is 5, 10, 15 for ANN; 18, 37, 56 for Cleveland
    //5,10,16,21,26,31,37 for acs_42(1/8 -7/8)
    //for final: 5 10 15 20 24 26 29 34 39
    //for 13 possible： 2 3 2	5	8	10 11
    //for 16 possible： 2 4 6 8 10 12 14
    //MACCE 73: 9 18 27 37 46 55 64
    //MACE_d14: 2 4 5 7 9 11 12
    //MACE_d36: 5 9 14 18 23 27 32

    val newFeatureSize = 5
    val selector = new ChiSqSelector()
      .setNumTopFeatures(newFeatureSize)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val model = selector.fit(data)

    //get selected DF
    val ChiSqDF = model.transform(data)
    val newChiDF = ChiSqDF.drop("features").withColumnRenamed("selectedFeatures", "features")
    newChiDF.printSchema()
    newChiDF.show()
    newChiDF.select("features").take(5).foreach(println)

    //Ｃonstruct RDD[Labeledpoint], in the form of row(label, vector), for LR
    val labeleddata = newChiDF.map(row => LabeledPoint(row.getAs[Double]("label"),
      row.getAs[org.apache.spark.mllib.linalg.Vector]("features"))
    )

    //save file, delete first if exists
    val libsvmFormPath = "processedData/" + dataname + ".Chisq." + newFeatureSize + ".libsvmForm"
    //    val libsvmFormPath = "processedData/" + dataname + ".Chisq." + newFeatureSize + "oversampling.libsvmForm"
    val libsvmFormFile = new File(libsvmFormPath)
    delete(libsvmFormFile)
    MLUtils.saveAsLibSVMFile(labeleddata, libsvmFormPath)

    ChiSqSelectedFeatureIndex.getSelectedFeatureIndex(model)

    sparkContext.stop()
  }

  private def delete(file: File) {
    if (file.isDirectory)
      Option(file.listFiles).map(_.toList).getOrElse(Nil).foreach(delete(_))
    file.delete
  }
}