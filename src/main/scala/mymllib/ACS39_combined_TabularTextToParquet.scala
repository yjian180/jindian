package mymllib

import java.io._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


object ACS39_combined_TabularTextToParquet {
  //output Parquet is DataFrame in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ACS39_combined_TabularTextToParquet").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger().setLevel(Level.ERROR)
    val sqlContext: SQLContext = new SQLContext(sparkContext)


    //Load data and StringType first, then assign the right DataTypefor each column
    val struct =
      StructType(
        StructField("Age", DoubleType, true) :: //DoubleType
          StructField("Sex", DoubleType, true) ::
          StructField("DM", DoubleType, true) ::
          StructField("HTA", DoubleType, true) ::
          StructField("DLP", DoubleType, true) ::
          StructField("PriorAMI", DoubleType, true) ::
          StructField("PriorPCI", DoubleType, true) ::
          StructField("PriorCABG", DoubleType, true) ::
          StructField("Hx_Stroke", DoubleType, true) ::
          StructField("Hx_CHF", DoubleType, true) ::
          StructField("CKD", StringType, true) :: //StringType
          StructField("Malignancy", DoubleType, true) ::
          StructField("SU", DoubleType, true) :: //combine STEMI and UA
          StructField("LVEF", DoubleType, true) :: //DoubleType
          StructField("Hb_Admission", DoubleType, true) :: //DoubleType
          StructField("Crea_Admission", DoubleType, true) :: //DoubleType
          StructField("Multivessel", DoubleType, true) ::
          StructField("DES", DoubleType, true) ::
          StructField("Thrombolysis", DoubleType, true) ::
          StructField("Revasc_Complete", StringType, true) :: //StringType
          StructField("CTP", DoubleType, true) :: //combine Clopi, Tica, Prasu
          StructField("OAC", DoubleType, true) ::
          StructField("BB", DoubleType, true) ::
          StructField("ACEI_ARB", DoubleType, true) ::
          StructField("Statin", DoubleType, true) ::
          StructField("PPI", DoubleType, true) ::
          StructField("Killip2", StringType, true) :: //y
          StructField("Transfusion_IH", StringType, true) :: //has null  y
          StructField("Bleeding_IH", DoubleType, true) ::
          StructField("PAD", DoubleType, true) ::
          StructField("HF_IH", StringType, true) :: //has null
          StructField("Continent", StringType, true) :: //StringType
          StructField("Country", StringType, true) :: //StringType
          StructField("Hx_Bleeding", DoubleType, true) ::
          StructField("PCI_whithoutStent", DoubleType, true) ::
          StructField("ReAMI_IH", StringType, true) :: //y
          StructField("Ulcus", StringType, true) :: //has null y
          StructField("Femoral_Access", DoubleType, true) ::
          StructField("AAS", DoubleType, true) ::
          StructField("label", DoubleType, true) :: Nil
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
    val dataname = "acs_39_combined"
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