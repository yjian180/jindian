package mymllib

import java.io._

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


object ANNTextToParquet {
  //output Parquet is DataFrame in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ANNTextToParquet").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sparkContext)


    //Load data and StringType first, then assign the right DataTypefor each column
    val struct =
      StructType(
        StructField("f1", DoubleType, true) ::
          StructField("f2", BooleanType, true) ::
          StructField("f3", BooleanType, true) ::
          StructField("f4", BooleanType, true) ::
          StructField("f5", BooleanType, true) ::
          StructField("f6", BooleanType, true) ::
          StructField("f7", BooleanType, true) ::
          StructField("f8", BooleanType, true) ::
          StructField("f9", BooleanType, true) ::
          StructField("f10", BooleanType, true) ::
          StructField("f11", BooleanType, true) ::
          StructField("f12", BooleanType, true) ::
          StructField("f13", BooleanType, true) ::
          StructField("f14", BooleanType, true) ::
          StructField("f15", BooleanType, true) ::
          StructField("f16", BooleanType, true) ::
          StructField("f17", DoubleType, true) ::
          StructField("f18", DoubleType, true) ::
          StructField("f19", DoubleType, true) ::
          StructField("f20", DoubleType, true) ::
          StructField("f21", DoubleType, true) ::
          StructField("label", StringType, true) :: Nil
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
    val dataname = "ann"
    val dataPath = "data/" + dataname + ".txt"
    val cleveland = sparkContext.textFile(dataPath)
    val rowRDD = cleveland.map(_.split("""\s+""")).map(p => Row.fromSeq(p.toSeq))
    var clevelandDataFrame = sqlContext.createDataFrame(rowRDD, orischema)

    clevelandDataFrame.dtypes.toList.foreach(println)
    clevelandDataFrame.take(3).foreach(println)
    clevelandDataFrame.printSchema()

    // Convert each column as its corresponding right type.
    for (i <- 0 until fnames.length) {
      val structfield = struct.fields.apply(i)
      if (structfield.dataType.toString.equals("StringType") || structfield.dataType.toString.equals("BooleanType")) {
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