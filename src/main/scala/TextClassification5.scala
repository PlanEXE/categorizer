import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
//import org.apache.spark.implicits._

//Word2Vec classification + random forest
//Test Accuracy = 0.6456858750820011
//Test Error = 0.35431412491799885
object TextClassification5 extends App {
  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .getOrCreate()

    import spark.implicits._

    val maxFeatures = 100

    val df = spark.read.json("/home/martin/dev/projects/scripts/ml-category-scrapper/*")
    val ds = df.select("title", "category").as[LeanItem3]

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    StopWordsRemover.loadDefaultStopWords("spanish")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("wordsFiltered")


    val word2vec = new Word2Vec()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
      .setWindowSize(3)
      .setMinCount(0)
      .setVectorSize(maxFeatures)

    val hashingPipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, word2vec))

    val featureDS = hashingPipeline.fit(ds).transform(ds)

    val labelIndexer = new StringIndexer()
      .setInputCol("cat_0")
//      .setInputCol("category")
      .setOutputCol("indexedLabel")
      .fit(featureDS)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(maxFeatures)
      .fit(featureDS)

    val preProcDS = featureIndexer.transform(labelIndexer.transform(featureDS))

    val Array(trainDS, testDS) = preProcDS.randomSplit(Array(0.9, 0.1), seed = 19)

    trainDS.show
    testDS.show

    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(100)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(rf, labelConverter))
    val model = pipeline.fit(trainDS)

    val predictions = model.transform(testDS)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)


    println(s"DataSet size: ${ds.count()}")
    println(s"TEST DataSet size: ${testDS.count()}")
    println(s"TEST DataSet size: ${testDS.select("title", "category").randomSplit(Array(1, 0)).head.take(100).mkString(", ")}")
    println(s"DataSet categories size: ${labelIndexer.labels.length}")
    println(s"DataSet categories: ${labelIndexer.labels.mkString(", ")}")
    predictions.select("title", "prediction", "indexedLabel", "predictedLabel", "category").show(500, truncate = false)
    println("Test Accuracy = " + accuracy)
    println("Test Error = " + (1.0 - accuracy))

  }
}