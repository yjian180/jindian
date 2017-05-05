name := "project_jindian"

version := "1.0"

scalaVersion := "2.10.6"


libraryDependencies ++= Seq(
  //  "org.eclipse.jetty" % "jetty-webapp" % "9.2.10.v20150310" % "container; compile",
  "javax.servlet" % "javax.servlet-api" % "3.1.0" % "compile",
  // Add Spark dependencies
  "org.apache.spark" % "spark-core_2.10" % "1.6.1" % "compile",
  "org.apache.spark" % "spark-mllib_2.10" % "1.6.1"% "compile",
  "org.apache.spark" % "spark-sql_2.10" % "1.6.1" % "compile",
  // Add Hadoop dependencies
  "org.apache.hadoop" % "hadoop-common" % "2.6.2" % "compile" excludeAll ExclusionRule(organization = "javax.servlet"),
  "org.apache.httpcomponents" % "httpclient" % "4.5.1" % "provided",
  "com.github.scopt" %% "scopt" % "3.4.0" % "provided"
)
    