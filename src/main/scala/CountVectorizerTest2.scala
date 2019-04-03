import ml.combust.mleap.spark.SparkSupport
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

//Naive bayes classfication + hashingTF
//Test set accuracy = 0.9507696051590923 => 1000000 features
//Test set accuracy = 0.9578104399628888 => 500000 features
//Test set accuracy = 0.9549639760654537 => 100000 features
//Test set accuracy = 0.9477678191392331 => 50000 features
//Test set accuracy = 0.8809176098332725 => 5000 features
//TODO: reduce features !

//Test set accuracy = 0.7897178702471406 => 1M Features - CAT_0
//Test set accuracy = 0.6884807772179786 - CAT_1 (Sin separar)

object CountVectorizerTest2 extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    import spark.implicits._

    //val df = spark.read.json("./cellphones/*").as[LeanItem3].take(20).toSeq.toDF.as[LeanItem3]

    val df = Seq(
      (Seq("a", "e"), "Vocales"),
      (Seq("a", "i"), "Vocales"),
      (Seq("a", "o"), "Vocales"),
      (Seq("a", "u"), "Vocales"),
      (Seq("b", "c"), "Consonantes"),
      (Seq("b", "d"), "Consonantes"),
      (Seq("b", "e"), "Consonantes"),
      (Seq("b", "f"), "Consonantes"),
      (Seq("b", "g"), "Consonantes"),
      (Seq("b", "h"), "Consonantes")

    ).toDF("words", "label")

    df.show(50, false)

    val vocabSize = 50


    val cvBinary = new CountVectorizer()
      .setBinary(true)
      .setVocabSize(vocabSize)
      .setInputCol("words")
      .setOutputCol("features")

    val cvBinaryModel = cvBinary.fit(df)

    println(cvBinaryModel.getVocabSize)
  }
}