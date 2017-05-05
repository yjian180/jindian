package mymllib

import java.io._

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


object TabularTextToParquet {
  //output Parquet is DataFrame in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("TabularTextToParquet").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)


    //Load data and StringType first, then assign the right DataTypefor each column
    val struct =
      StructType(
        StructField("id", DoubleType, true) ::
          StructField("ccf", DoubleType, true) ::
          StructField("age", DoubleType, true) ::
          StructField("sex", DoubleType, true) ::
          StructField("painloc", DoubleType, true) ::
          StructField("painexer", DoubleType, true) ::
          StructField("relrest", DoubleType, true) ::
          StructField("pncaden", DoubleType, true) ::
          StructField("cp", DoubleType, true) ::
          StructField("trestbps", DoubleType, true) ::
          StructField("htn", DoubleType, true) ::
          StructField("chol", DoubleType, true) ::
          StructField("smoke", DoubleType, true) ::
          StructField("cigs", DoubleType, true) ::
          StructField("years", DoubleType, true) ::
          StructField("fbs", DoubleType, true) ::
          StructField("dm", DoubleType, true) ::
          StructField("famhist", DoubleType, true) ::
          StructField("restecg", DoubleType, true) ::
          StructField("ekgmo", DoubleType, true) ::
          StructField("ekgday", DoubleType, true) ::
          StructField("ekgyr", DoubleType, true) ::
          StructField("dig", DoubleType, true) ::
          StructField("prop", DoubleType, true) ::
          StructField("nitr", DoubleType, true) ::
          StructField("pro", DoubleType, true) ::
          StructField("diuretic", DoubleType, true) ::
          StructField("proto", DoubleType, true) ::
          StructField("thaldur", DoubleType, true) ::
          StructField("thaltime", DoubleType, true) ::
          StructField("met", DoubleType, true) ::
          StructField("thalach", DoubleType, true) ::
          StructField("thalrest", DoubleType, true) ::
          StructField("tpeakbps", DoubleType, true) ::
          StructField("tpeakbpd", DoubleType, true) ::
          StructField("dum", DoubleType, true) ::
          StructField("trestbpd", DoubleType, true) ::
          StructField("exang", DoubleType, true) ::
          StructField("xhypo", DoubleType, true) ::
          StructField("oldpeak", DoubleType, true) ::
          StructField("slope", DoubleType, true) ::
          StructField("rldv5", DoubleType, true) ::
          StructField("rldv5e", DoubleType, true) ::
          StructField("ca", DoubleType, true) ::
          StructField("restckm", DoubleType, true) ::
          StructField("exerckm", DoubleType, true) ::
          StructField("restef", DoubleType, true) ::
          StructField("restwm", DoubleType, true) ::
          StructField("exeref", DoubleType, true) ::
          StructField("exerwm", DoubleType, true) ::
          StructField("thal", DoubleType, true) ::
          StructField("thalsev", DoubleType, true) ::
          StructField("thalpul", DoubleType, true) ::
          StructField("earlobe", DoubleType, true) ::
          StructField("cmo", DoubleType, true) ::
          StructField("cday", DoubleType, true) ::
          StructField("cyr", DoubleType, true) ::
          StructField("label", DoubleType, true) ::
          StructField("lmt", DoubleType, true) ::
          StructField("ladprox", DoubleType, true) ::
          StructField("laddist", DoubleType, true) ::
          StructField("diag", DoubleType, true) ::
          StructField("cxmain", DoubleType, true) ::
          StructField("ramus", DoubleType, true) ::
          StructField("om1", DoubleType, true) ::
          StructField("om2", DoubleType, true) ::
          StructField("rcaprox", DoubleType, true) ::
          StructField("rcadist", DoubleType, true) ::
          StructField("lvx1", DoubleType, true) ::
          StructField("lvx2", DoubleType, true) ::
          StructField("lvx3", DoubleType, true) ::
          StructField("lvx4", DoubleType, true) ::
          StructField("lvf", DoubleType, true) ::
          StructField("cathef", DoubleType, true) ::
          StructField("junk", DoubleType, true) ::
          StructField("name", StringType, true) :: Nil
      )
    val fnames = struct.fieldNames
    var schemaString = ""
    for (i <- 0 until fnames.length - 1) {
      val tmpstructfield = struct.fields.apply(i)
      schemaString = schemaString + tmpstructfield.name + " "
    }
    schemaString = schemaString + struct.fields.apply(fnames.length - 1).name
    val orischema =
      StructType(
        schemaString.split(" ").map(fieldName => StructField(fieldName.trim(), StringType, true)))

    //Load as dataFrame. Columns are all StringType
    val dataname = "cleveland"
    val dataPath = "data/" + dataname + "_dataset.txt"
    val cleveland = sparkContext.textFile(dataPath)
    val rowRDD = cleveland.map(_.split("""\s+""")).map(p => Row.fromSeq(p.toSeq))
    var clevelandDataFrame = sqlContext.createDataFrame(rowRDD, orischema)


    clevelandDataFrame.dtypes.toList.foreach(println)
    clevelandDataFrame.take(3).foreach(println)
    clevelandDataFrame.printSchema()


    // Convert each column as its corresponding right type.
    for (i <- 0 until fnames.length) {
      val structfield = struct.fields.apply(i)
      if (structfield.dataType.toString.equals("StringType")) {
        //index String type features
        val fname = structfield.name
        val indexer = new StringIndexer()
          .setInputCol(fname)
          .setOutputCol("tmp" + fname)
        clevelandDataFrame = indexer.fit(clevelandDataFrame).transform(clevelandDataFrame)
        clevelandDataFrame = clevelandDataFrame.drop(fname).withColumnRenamed("tmp" + fname, fname)
      }
      else {
        //convert numeric feature to its corresponding type according to schema
        clevelandDataFrame = clevelandDataFrame.withColumn(
          structfield.name, clevelandDataFrame(structfield.name).cast(DoubleType))
      }
    }
    clevelandDataFrame.printSchema()

    //save parquet file, delete first if exists
    val parquetPath = "processedData/" + dataname + ".parquet"
    val parquetFile = new File(parquetPath)
    delete(parquetFile)
    clevelandDataFrame.write.parquet(parquetPath)

    //output total number and each class
    println(clevelandDataFrame.select("label").map(a => (a, 1)).reduceByKey(_ + _).getClass)
    clevelandDataFrame.select("label").map(a => (a, 1)).reduceByKey(_ + _).foreach(println)
    println("count:" + clevelandDataFrame.count())

    sparkContext.stop()
  }

  private def delete(file: File) {
    if (file.isDirectory)
      Option(file.listFiles).map(_.toList).getOrElse(Nil).foreach(delete(_))
    file.delete
  }

}