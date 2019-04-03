import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, NaiveBayes, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

//Random forest + hashingTF
object TextClassification2 extends App {
  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .getOrCreate()

    import spark.implicits._

    val maxFeatures = 10000

    val ds = spark.read.json("/home/martin/dev/projects/scripts/ml-category-scrapper/*").as[LeanItem3]

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    StopWordsRemover.loadDefaultStopWords("spanish")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("wordsFiltered")

    val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
      .setNumFeatures(maxFeatures)

    val hashingPipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF))

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

//    val preProcPipeline = new Pipeline()
//      .setStages(Array(labelIndexer, featureIndexer))

    val preProcDS = featureIndexer.transform(labelIndexer.transform(featureDS))

    val Array(trainDS, testDS) = preProcDS.randomSplit(Array(0.9, 0.1), seed = 19)

    trainDS.show
    testDS.show

    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(8)
//      .setMaxBins(100)
      .setNumTrees(50)

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

    pipeline.save("./myRFModel-8l-50t-10000f")
    println(s"DataSet size: ${ds.count()}")
    println(s"TEST DataSet size: ${testDS.count()}")
    println(s"TEST DataSet size: ${testDS.select("title", "category").randomSplit(Array(1, 0)).head.take(100).mkString(", ")}")
    println(s"DataSet categories size: ${labelIndexer.labels.length}")
    println(s"DataSet categories: ${labelIndexer.labels.mkString(", ")}")
    println("Test Accuracy = " + accuracy)
    println("Test Error = " + (1.0 - accuracy))
    predictions.select("title", "prediction", "indexedLabel", "predictedLabel", "cat_0").show(500, truncate = false)
  }
}