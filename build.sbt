name := "SparkML"

version := "1.0"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.2.1" % "provided",
  "org.apache.spark" % "spark-sql_2.11" % "2.2.1" % "provided",
  "org.apache.spark" % "spark-streaming_2.11" % "2.2.1" % "provided",
  "org.apache.spark" % "spark-mllib_2.11" % "2.2.1" % "provided",
  "ml.combust.mleap" %% "mleap-spark" % "0.9.6",
  "com.amazonaws" % "aws-java-sdk" % "1.7.4",
  "org.apache.hadoop" % "hadoop-aws" % "2.7.1",
  "net.java.dev.jets3t" % "jets3t" % "0.9.0",
//  "edu.stanford.nlp" % "stanford-corenlp" % "3.9.1",
"com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
//  "org.apache.spark" % "spark-core_2.11" % "2.2.1",
//  "org.apache.spark" % "spark-sql_2.11" % "2.2.1",
//  "org.apache.spark" % "spark-streaming_2.11" % "2.2.1",
//  "org.apache.spark" % "spark-mllib_2.11" % "2.2.1"
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case n if n.startsWith("reference.conf") => MergeStrategy.concat
  case x => MergeStrategy.first
}