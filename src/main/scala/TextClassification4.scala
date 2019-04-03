import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, RegexTokenizer, StringIndexer, Word2Vec}
import org.apache.spark.sql.SparkSession

//Word2Vec classification
object TextClassification4 extends App {
  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .getOrCreate()

    import spark.implicits._

    val df = spark.read.json("/home/martin/dev/projects/scripts/ml-category-scrapper/*")

    val ds = df.as[LeanItem3]

    val labelIndexer = new StringIndexer()
      .setInputCol("cat_0")
      .setOutputCol("label")

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    val word2Vec = new Word2Vec()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
      .setWindowSize(5)
      .setVectorSize(1000)
      .setMinCount(0)

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, word2Vec, scaler))

    import spark.implicits._

    val dsLabeled = pipeline.fit(ds).transform(ds)

    dsLabeled.show()

    val Array(trainDF, testDF) = dsLabeled.randomSplit(Array(0.9, 0.1))

    val naiveBayesModel = new NaiveBayes().setSmoothing(0.2).setFeaturesCol("scaledFeatures").setLabelCol("label").fit(trainDF)

    //naiveBayesModel.transform(hashed).show()
    val predictions = naiveBayesModel.transform(testDF)
    //predictions.map{ case Row(t: String,l,c: String,w,f,rp,pr,p: Double) => (t, c, categoriesByIndex(p.toInt)(0)._1)}.show(100, false)


    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
        println("Test set accuracy = " + accuracy)

  }
}