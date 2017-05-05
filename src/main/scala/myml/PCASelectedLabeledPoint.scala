package myml

import java.io.File

import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object PCASelectedLabeledPoint {
  //output LabeledPoint is RDD[LabeledPoint] in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("PCASelectedLabeledPoint").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)

    //specify dataname: cleveland or ann
    val dataname = "cleveland"
    //    val dataname = "ann"

    // Load the data stored in LIBSVM format as a DataFrame. Specify datatype: original data or oversampling data.
    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + ".libsvmForm")
    //    val data = sqlContext.read.format("libsvm").load("processedData/" + dataname + "oversampling.libsvmForm")

    //Features ize should be less than 21 and 75 for ann and cleveland
    //We set it as 3/4, 1/2, 1/4 of original size, which is 5, 10, 15 for ANN; 18, 37, 56 for Cleveland
    val newFeatureSize = 15
    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(newFeatureSize)
      .fit(data)

    //get Transformed PCA DF
    val pcaDF = pca.transform(data)
    val newPCADF = pcaDF.drop("features").withColumnRenamed("pcaFeatures", "features")
    newPCADF.show()
    newPCADF.printSchema()
//    newPCADF.select("pcaFeatures").take(5).foreach(println)

    //Ｃonstruct RDD[Labeledpoint], in the form of row(label, vector), for LR
    val labeleddata = newPCADF.map(row => LabeledPoint(row.getAs[Double]("label"),
      row.getAs[org.apache.spark.mllib.linalg.Vector]("features"))
    )

    //save parquet file, delete first if exists
    val libsvmFormPath = "processedData/" + dataname + ".PCA." + newFeatureSize + ".libsvmForm"
    //    val libsvmFormPath = "processedData/" + dataname + ".PCA." + newFeatureSize + "oversampling.libsvmForm"
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