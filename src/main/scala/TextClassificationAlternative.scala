import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.sql.SparkSession
import resource._

object TextClassificationAlternative extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    import spark.implicits._

    val ds = spark.read.json("./dataset/*").as[LeanItem3]


    def savePipeline(name: String, transoformers: Transformer*): Unit = {
      val sbc = SparkBundleContext()
      for(bf <- managed(BundleFile(s"jar:file:/home/martin/dev/projects/SparkML/results/$name.zip"))) {
        SparkUtil.createPipelineModel(transoformers.toArray).writeBundle.save(bf)(sbc).get
      }
    }

    //--------------PRE-PROCESS-------------

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    StopWordsRemover.loadDefaultStopWords("spanish")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("wordsFiltered")

    //    val cv = new CountVectorizer()
    //      .setInputCol(remover.getOutputCol)
    //      .setOutputCol("features")
    //      .setVocabSize(50000)

    val cv =  new HashingTF()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
      .setNumFeatures(50000)

    val preProcessPipeline = new Pipeline().setStages(Array(tokenizer, remover, cv)).fit(ds)

    //savePipeline("preProcess", preProcessPipeline)

    //--------------PRE-PROCESS-------------

    val preProcessedItems = preProcessPipeline.transform(ds).as[LeanItem3]

    val labelIndexer = new StringIndexer()
      .setInputCol("cat_2")
      .setOutputCol("label")
      .fit(preProcessedItems)

    val converter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("categories")
      .setLabels(labelIndexer.labels)

    //    savePipeline(s"converter-cat_0", converter)

    val labeledItems = labelIndexer.transform(preProcessedItems).as[LeanItem3]
    val Array(trainDF, testDF) = labeledItems.randomSplit(Array(0.9, 0.1))

    val naiveBayesModel = new NaiveBayes().setFeaturesCol("features").setLabelCol("label").fit(trainDF)

    //savePipeline(s"cat_0", naiveBayesModel, converter)
    println(s"cat_2")


    naiveBayesModel.transform(testDF).show()

    val predictions = naiveBayesModel.transform(testDF)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)

    //    println(s"Amount of categories in cat_0: ${ds.groupByKey(x => x.cat_0).keys.count()}")
    //    println(s"Amount of categories in cat_1: ${ds.groupByKey(x => x.cat_1).keys.count()}")
    //    println(s"Amount of categories in cat_1: ${ds.groupByKey(x => x.cat_2).keys.count()}")

    //    naiveBayesModel.save("./myNaiveBayesModel-1M-features")

  }
}
