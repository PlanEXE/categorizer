import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
//import org.apache.spark.implicits._

object NGram extends App {
  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .getOrCreate()

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    val ngramAssembler = new VectorAssembler()
//      .setInputCols(Array("ngrams1-features", "ngrams2-features", "ngrams3-features"))
      .setInputCols(Array("ngrams2-features", "ngrams1-features"))
      .setOutputCol("features")

    import spark.implicits._

    val ds = spark.read.json("/home/martin/dev/projects/scripts/ml-item-category/*").as[LeanItem2]

    val labelIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("label")

    val hashingTF = (col: String) => {
      new HashingTF()
        .setInputCol(col)
        .setOutputCol(s"$col-features")
        .setNumFeatures(20000)
    }

    val ngram = new NGram().setN(1).setInputCol("words").setOutputCol("ngrams1")
    val ngram2 = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams2")
//    val ngram3 = new NGram().setN(3).setInputCol("words").setOutputCol("ngrams3")

    val pipeline = new Pipeline().setStages(Array(
      labelIndexer,
      tokenizer,
      ngram2, ngram,
      hashingTF(ngram.getOutputCol),
      hashingTF(ngram2.getOutputCol),
//      hashingTF(ngram3.getOutputCol),
      ngramAssembler))


    val dsLabeled = pipeline.fit(ds).transform(ds)

    val Array(trainDF, testDF) = dsLabeled.randomSplit(Array(0.9, 0.1))

    val naiveBayesModel = new NaiveBayes().setFeaturesCol("features").setLabelCol("label").fit(trainDF)
    val predictions = naiveBayesModel.transform(testDF)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)
  }
}