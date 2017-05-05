package mymllib

import java.io._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object MACCE_d36_TabularTextToParquet {
  //output Parquet is DataFrame in essence

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("PCITabularTextToParquet").setMaster("local[2]")
    val sparkContext: SparkContext = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger().setLevel(Level.ERROR)
    val sqlContext: SQLContext = new SQLContext(sparkContext)


    //Load data and StringType first, then assign the right DataTypefor each column
    val struct =
      StructType(



        StructField("男性", StringType, true) ::
          StructField("年龄", DoubleType, true) ::
          StructField("BMI（入院时）", DoubleType, true) ::
          StructField("有无DM", StringType, true) :: //S
          StructField("有无HT", StringType, true) ::
          StructField("有无脑血管病史", StringType, true) ::
          StructField("有无OMI", StringType, true) ::
          StructField("有无重建史", StringType, true) ::
          StructField("有无吸烟史", StringType, true) ::
          StructField("诊断", StringType, true) ::
          StructField("稳定型CAD", StringType, true) ::
          StructField("SBP", DoubleType, true) ::
          StructField("DBP", DoubleType, true) ::
          StructField("房颤扑", StringType, true) ::
          StructField("LVEF", DoubleType, true) ::
          StructField("HBG", DoubleType, true) ::
          StructField("WBC", DoubleType, true) ::
          StructField("中性", DoubleType, true) ::
          StructField("BUN", DoubleType, true) ::
          StructField("CR", DoubleType, true) ::
          StructField("TC", DoubleType, true) ::
          StructField("TG", DoubleType, true) ::
          StructField("LDL", DoubleType, true) ::
          StructField("HDL", DoubleType, true) ::
          StructField("空腹血糖", DoubleType, true) ::
          StructField("病变支数", DoubleType, true) :: //S no
          StructField("左主干病变", StringType, true) ::
          StructField("重建类型", StringType, true) ::
          StructField("有无开口病变", StringType, true) ::
          StructField("有无LAD近端病变", StringType, true) ::
          StructField("有无LM病变", StringType, true) ::
          StructField("住院是否使用ASA", StringType, true) ::
          StructField("住院是否使用β阻滞剂", StringType, true) ::
          StructField("住院是否使用钙拮抗剂", StringType, true) ::
          StructField("住院硝酸酯", StringType, true) ::
          StructField("住院是否使用ACEI", StringType, true) ::
          StructField("label", DoubleType, true) :: Nil
//                  StructField("是否发生随访MACCE", StringType, true) ::
//                  StructField("随访MACCE至血运重建天", DoubleType, true) ::
//                  StructField("是否发生总MACCE", StringType, true) :: Nil
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
    val dataname = "MACCE_d36"
    val dataPath = "data/" + dataname + ".txt"
    val cleveland = sparkContext.textFile(dataPath)
    val rowRDD = cleveland.map(_.split("""\s+""")).map(p => Row.fromSeq(p.toSeq))
    var clevelandDataFrame = sqlContext.createDataFrame(rowRDD, orischema)

//    clevelandDataFrame.dtypes.toList.foreach(println)
//    clevelandDataFrame.take(3).foreach(println)
//    clevelandDataFrame.printSchema()


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
//    clevelandDataFrame.printSchema()

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