import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import resource._

//Naive bayes classfication + hashingTF
//Test set accuracy = 0.9507696051590923 => 1000000 features
//Test set accuracy = 0.9578104399628888 => 500000 features
//Test set accuracy = 0.9549639760654537 => 100000 features
//Test set accuracy = 0.9477678191392331 => 50000 features
//Test set accuracy = 0.8809176098332725 => 5000 features
//TODO: reduce features !

//Test set accuracy = 0.7897178702471406 => 1M Features - CAT_0
//Test set accuracy = 0.6884807772179786 - CAT_1 (Sin separar)

object CountVectorizerTest extends App with SparkSupport {
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
      (Seq("celular", "samsung", "galaxy", "s6"), "Celular"),
      (Seq("celular", "lg"), "Celular"),
      (Seq("celular", "galaxy", "s6"), "Celular"),
      (Seq("celular", "galaxy"), "Celular"),
      (Seq("s6"), "Celular"),
      (Seq("galaxy", "s6"), "Celular"),
      (Seq("samsung", "s6"), "Celular"),
      (Seq("funda", "samsung", "galaxy", "s6"), "Fundas"),
      (Seq("funda", "celular", "samsung"), "Fundas"),
      (Seq("funda", "samsung"), "Fundas"),
      (Seq("funda", "samsung", "iphone"), "Fundas"),
      (Seq("funda", "celular"), "Fundas"),
      (Seq("funda", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "samsung"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "varios", "disenios", "samsung", "s6"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "samsung", "galaxy", "s6"), "Fundas"),
      (Seq("funda", "celular", "samsung"), "Fundas"),
      (Seq("funda", "samsung"), "Fundas"),
      (Seq("funda", "samsung", "iphone"), "Fundas"),
      (Seq("funda", "celular"), "Fundas"),
      (Seq("funda", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "samsung"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "varios", "disenios", "samsung", "s6"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "samsung", "galaxy", "s6"), "Fundas"),
      (Seq("funda", "celular", "samsung"), "Fundas"),
      (Seq("funda", "samsung"), "Fundas"),
      (Seq("funda", "samsung", "iphone"), "Fundas"),
      (Seq("funda", "celular"), "Fundas"),
      (Seq("funda", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "samsung"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "varios", "disenios", "samsung", "s6"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "samsung", "galaxy", "s6"), "Fundas"),
      (Seq("funda", "celular", "samsung"), "Fundas"),
      (Seq("funda", "samsung"), "Fundas"),
      (Seq("funda", "samsung", "iphone"), "Fundas"),
      (Seq("funda", "celular"), "Fundas"),
      (Seq("funda", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "samsung"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "varios", "disenios", "samsung", "s6"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "samsung", "galaxy", "s6"), "Fundas"),
      (Seq("funda", "celular", "samsung"), "Fundas"),
      (Seq("funda", "samsung"), "Fundas"),
      (Seq("funda", "samsung", "iphone"), "Fundas"),
      (Seq("funda", "celular"), "Fundas"),
      (Seq("funda", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "samsung"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "varios", "disenios", "samsung", "s6"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "samsung", "galaxy", "s6"), "Fundas"),
      (Seq("funda", "celular", "samsung"), "Fundas"),
      (Seq("funda", "samsung"), "Fundas"),
      (Seq("funda", "samsung", "iphone"), "Fundas"),
      (Seq("funda", "celular"), "Fundas"),
      (Seq("funda", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "samsung"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "varios", "disenios", "samsung", "s6"), "Fundas"),
      (Seq("funda", "silicona", "iphone"), "Fundas"),
      (Seq("funda", "silicona", "iphone", "samsung"), "Fundas")
    ).toDF("words", "label")
    //    val trainDF = train.toDF("text")

    df.show(50, false)

    //    val vocabSize = train.length
    //
    //    val cv = new CountVectorizer()
    //        .setBinary(true)
    //      .setVocabSize(vocabSize)
    //      .setInputCol("text")
    //
    //    val cvModel = cv.fit(trainDF)
    //
    //    println(cvModel.vocabulary.mkString(", "))
    //
    //    cvModel.transform(test).show(50, false)

//    val tokenizer = new Tokenizer()
//      .setInputCol("title")
//      .setOutputCol("words")
//
//    val tokenized = tokenizer.transform(df)

    val vocabSize = 50


    val cvBinary = new CountVectorizer()
      .setBinary(true)
      .setVocabSize(vocabSize)
      .setInputCol("words")
      .setOutputCol("features")

    val cvBinaryModel = cvBinary.fit(df)

    val cvBinaryProcessed = cvBinaryModel.transform(df)

    cvBinaryProcessed.show(50, false)

    println("_____________________________________________________________________")

    val si = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexed_label")

    val indexed = si.fit(df).transform(cvBinaryProcessed)

    new NaiveBayes().setLabelCol("indexed_label").setModelType("bernoulli").fit(indexed).transform(indexed).show(50, false)

    //    val indexer = new StringIndexer()
    //      .setInputCol("words")
    //      .setOutputCol("categoryIndex")
    //      .fit(tokenized)
    //
    //    val indexed = indexer.transform(tokenized)
    //
    //    val encoder = new OneHotEncoder()
    //      .setInputCol("categoryIndex")
    //      .setOutputCol("categoryVec")
    //
    //    val encoded = encoder.transform(indexed)
    //    encoded.show()
  }
}