import java.text.Normalizer

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import resource._

import scala.collection.parallel.ForkJoinTaskSupport
import scala.reflect.ClassTag

object TextClassificationTests3 extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    implicit val sparkSession: SparkSession = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    val sc = sparkSession.sparkContext

    import sparkSession.implicits._

    def normalizeText(ds: Dataset[LeanItem3])(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      val rInches = "(?<=^|\\s)(\\d+)('|''|\")($|\\s|\\.$|\\.\\s)"
      val rMm = "(?<=^|\\s|x)(\\d+) ?(mm|milimetro|milimetros)($|\\s|\\.$|\\.\\s|!|%)" //Puede empezar con x ej: 20x40cm -> 20x40 centimetros
      val rCm = "(?<=^|\\s|x)(\\d+) ?(cm|centimetros|centimetro)($|\\s|\\.$|\\.\\s|!|%)"
      val rCM = "(?<=^|\\s|x)(\\d+) ?(m|metro|metros)($|\\s|\\.$|\\.\\s|!|%)"
      val rLts = "(?<=^|\\s)(\\d+) ?(lts|l|litros|litro)($|\\s|\\.$|\\.\\s|!|%)"
      val rMLts = "(?<=^|\\s|x)(\\d+) ?(ml|mililitros|mililitro)($|\\s|\\.$|\\.\\s|!|%)"
      val rGrams = "(?<=^|\\s)(\\d+) ?(gr|grs|gramos|gramo)($|\\s|\\.$|\\.\\s|!|%)" //No usamos la G por 4g por ejemplo
      val rKilo = "(?<=^|\\s)(\\d+) ?(kilo|kilos)($|\\s|\\.$|\\.\\s|!|%)" //No usamos la K por Tv 4K por ejemplo
      val rWatts = "(?<=^|\\s)(\\d+) ?(w|watt|watts)($|\\s|\\.$|\\.\\s|!|%)"
      val rVolts = "(?<=^|\\s)(\\d+) ?(v|volt|volts)($|\\s|\\.$|\\.\\s|!|%)"
      val rNumbers = "(?<=^|\\s)(\\d+|\\d+\\.\\d+|\\d+,\\d+)(?=$|\\s|\\.$|\\.\\s|,\\s|%|!|%)"
      val rNumbers2 = "(?<=^|\\s|,|\\.)(\\d+)(?=$|\\s|\\.$|\\.\\s|,|%|!|%)" //ej: 1,2,3,4
      val rNumbers3 = "(?<=^|\\s)(\\d+x\\d+)(?=$|\\s|\\.$|\\.\\s|,\\s|%|!|%)" //Soluciona los 20x30
      val rAmpersand = "&"
      val rPercentage = "%"
      val rPunctuation = "[^\\w\\s\\+'\"/]"
      val rPunctuation2 = "\\s/\\s"
      val rSpaces = "\\s+"


      ds.map(x => {
        val newTitle = Normalizer.normalize(x.title.toLowerCase.trim, Normalizer.Form.NFD).replaceAll("[^\\p{ASCII}]", "")
          .replaceAll(rInches, "$1 pulgadas$3")
          .replaceAll(rMm, "$1 milimetros$3")
          .replaceAll(rCm, "$1 centimetros$3")
          .replaceAll(rCM, "$1 metros$3")
          .replaceAll(rLts, "$1 litros$3")
          .replaceAll(rMLts, "$1 mililitros$3")
          .replaceAll(rGrams, "$1 gramos$3")
          .replaceAll(rKilo, "$1 kilogramos$3")
          .replaceAll(rWatts, "$1 watts$3")
          .replaceAll(rVolts, "$1 volts$3")
          .replaceAll(rNumbers3, "NUMBER x NUMBER")
          .replaceAll(rNumbers2, "NUMBER")
          .replaceAll(rNumbers, "NUMBER")
          .replaceAll(rAmpersand, " y ")
          .replaceAll(rPercentage, " porciento ")
          .replaceAll(rPunctuation, " ")
          .replaceAll(rPunctuation2, " ")
          .replaceAll(rSpaces, " ")

        if (x.title != newTitle) x.copy(title = newTitle) else x
      })
    }

    val ds = normalizeText(sparkSession.read.json("./dataset/*").repartition(4).as[LeanItem3])

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    val ngram2 = new NGram()
      .setN(2)
      .setInputCol("words")
      .setOutputCol("2gram")

    val ngram3 = new NGram()
      .setN(3)
      .setInputCol("words")
      .setOutputCol("3gram")

    val tokenize = new Pipeline().setStages(Array(tokenizer, ngram2, ngram3)).fit(ds)

    val tokenized  = tokenize.transform(ds)

    val va = new VectorAssembler()
      .setInputCols(Array("featuresWords", "featuresGram2", "featuresGram3"))
      .setOutputCol("features")

    val cv = new CountVectorizer()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("featuresWords")
      .setBinary(true)
      .setVocabSize(60000)
      .setMinDF(100) //Con 100 no supera las 37K palabras

    val cvGram2 = new CountVectorizer()
      .setInputCol(ngram2.getOutputCol)
      .setOutputCol("featuresGram2")
      .setBinary(true)
      .setVocabSize(30000)
      .setMinDF(50)

    val cvGram3 = new CountVectorizer()
      .setInputCol(ngram3.getOutputCol)
      .setOutputCol("featuresGram3")
      .setBinary(true)
      .setVocabSize(30000)
      .setMinDF(50)

    val vcs = Array(cv, cvGram2, cvGram3)

    println(vcs.map(_.fit(tokenized).vocabulary.length).sum)
  }
}
