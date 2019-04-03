import java.text.Normalizer

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import resource._

import scala.reflect.ClassTag
import scala.util.Try
import scala.util.matching.Regex

object TextClassificationChiSq extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //    implicit val sparkSession = SparkSession.builder
    //      .master("local[*]")
    //      .appName("Spark Word Count")
    //      .config("spark.driver.maxResultSize", "10g")
    //      .getOrCreate()

    implicit val sparkSession = SparkSession.builder
      .appName("Text Classification ChiSq 01")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    import sparkSession.implicits._

    //    val ds = normalizeText(sparkSession.read.json("./cellphones/*").as[LeanItem3]).as[LeanItem3]
    //    val ds = normalizeText(sparkSession.read.json("/home/ubuntu/SparkML/dataset/*").repartition(144*2).as[LeanItem3]).as[LeanItem3]
    sparkSession.sparkContext.hadoopConfiguration.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sparkSession.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", "AKIAJ5HUWSP2TUXBV3OA")
    sparkSession.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", "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")
    val ds = normalizeText(sparkSession.read.json("s3a://wolla.datasets/ml-dataset/dataset/*").repartition(752*3).as[LeanItem3]).as[LeanItem3]

    def savePipeline(name: String, sparkDataframe: DataFrame, transformers: Transformer*): Unit = {
      val transformedDataset = transformers.foldLeft(sparkDataframe)((x,y) => y.transform(x))
      val fullPath = s"jar:file:/home/ubuntu/SparkML/results/${name.trim}.zip"
      new Pipeline().setStages(transformers.toArray).write.overwrite().save(s"/home/ubuntu/SparkML/results/${name.trim}-SPARK")
      //      val fullPath = s"jar:file:/home/martin/dev/wolla/SparkML/results/${name.trim}.zip"
      println(fullPath)
      implicit val sbc: SparkBundleContext = SparkBundleContext().withDataset(transformedDataset)
      //            for(bf <- managed(BundleFile(s"jar:file:/home/martin/dev/wolla/SparkML/results/${name.trim}.zip"))) {
      for(bf <- managed(BundleFile(fullPath))) {
        SparkUtil.createPipelineModel(transformers.toArray).writeBundle.save(bf)(sbc).get
      }
    }

    //--------------PRE-PROCESS-------------

    val tokenizer = new RegexTokenizer().setInputCol("title").setOutputCol("words")
    val ngram2 = new NGram().setN(2).setInputCol("words").setOutputCol("2gram")
    val ngram3 = new NGram().setN(3).setInputCol("words").setOutputCol("3gram")

    def assembleVectors(cvs: Array[CountVectorizer]) =  new VectorAssembler().setInputCols(cvs.map(_.getOutputCol)).setOutputCol("featuresAssembled")

    def createVectorizers() = {
      val cv = new CountVectorizer()
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol("featuresWords")
        .setBinary(true)
        .setVocabSize(300000)
        .setMinDF(50)

      val cvGram2 = new CountVectorizer()
        .setInputCol(ngram2.getOutputCol)
        .setOutputCol("featuresGram2")
        .setBinary(true)
        .setVocabSize(100000)
        .setMinDF(50)

      val cvGram3 = new CountVectorizer()
        .setInputCol(ngram3.getOutputCol)
        .setOutputCol("featuresGram3")
        .setBinary(true)
        .setVocabSize(100000)
        .setMinDF(50)

      Array(cv, cvGram2, cvGram3)
    }

    val labelers = (0 to 2).map(i => new StringIndexer().setInputCol(s"cat_$i").setOutputCol(s"label_cat_$i").fit(ds)).toArray

    val labeledItems = labelers.foldLeft(ds)((data, model) => model.transform(data).as[LeanItem3]).cache()

    val transformStages = Array(tokenizer, ngram2, ngram3)
    val vectorizers = createVectorizers()
    val featuresAssembler = assembleVectors(vectorizers)

    val selector = new ChiSqSelector().setFeaturesCol(featuresAssembler.getOutputCol).setOutputCol("features").setNumTopFeatures(30000).setLabelCol("label_cat_2")

    val preProcessPipeline = new Pipeline().setStages(transformStages ++ vectorizers :+ featuresAssembler :+ selector).fit(labeledItems)

    labeledItems.unpersist()

    val preProcessedItems = preProcessPipeline.transform(ds).as[LeanItem3].cache()

    savePipeline("preProcess", ds.toDF(), preProcessPipeline)

    //--------------PRE-PROCESS-------------

    def trainBayes(f: LeanItem3 => String, category_column: String, labeler: StringIndexerModel): Unit = {

      val g = preProcessedItems.groupByKey(f).keys.collect()

      println(s"Groups: ${g.length} for: $category_column")

      val converter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("categories")
        .setLabels(labeler.labels)

      g.foreach(cat_n => {
        val itemDs = filterByPercentile3(preProcessedItems.filter((x) => f(x) == cat_n), 5, 95)

        val naiveBayesModel = new NaiveBayes().setModelType("bernoulli").setFeaturesCol("features").setLabelCol(s"label_$category_column").fit(itemDs)

        savePipeline(s"$cat_n-$category_column", itemDs.toDF(), naiveBayesModel, converter)

        println(s"$cat_n-$category_column")
      })
    }

    def trainLogReg(f: LeanItem3 => String, category_column: String, labeler: StringIndexerModel): Unit = {

      val g = preProcessedItems.groupByKey(f).keys.collect()

      println(s"Groups: ${g.length} for: $category_column")

      val converter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("categories")
        .setLabels(labeler.labels)

      g.foreach(cat_n => {
        val itemDs = filterByPercentile3(preProcessedItems.filter((x) => f(x) == cat_n), 5, 95).cache()

        val logRegModel = new LogisticRegression().setFeaturesCol("features").setLabelCol(s"label_$category_column").fit(itemDs)

        itemDs.unpersist()

        savePipeline(s"$cat_n-$category_column", itemDs.toDF(), logRegModel, converter)

        println(s"$cat_n-$category_column")
      })
    }

    val converter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("categories")
      .setLabels(labelers(0).labels)

//    val naiveBayesModel = new NaiveBayes().setModelType("bernoulli").setFeaturesCol("features").setLabelCol("label_cat_0").fit(labeledItems)
    val itemDs = filterByPercentile3(preProcessedItems, 2, 98).cache()
    val logRegModel = new LogisticRegression().setFeaturesCol("features").setLabelCol("label_cat_0").fit(itemDs)
    itemDs.unpersist()

    savePipeline(s"cat_0", itemDs.toDF(), logRegModel, converter)
    println(s"cat_0")

//    trainBayes((x) => x.cat_0, "cat_1", labelers(1))
    trainLogReg(_.cat_0, "cat_1", labelers(1))
    trainLogReg(_.cat_1, "cat_2", labelers(2))

    sparkSession.stop()


  }


  /**
    * compute percentile from an unsorted Spark RDD
    * @param data: input data set of Long integers, must be sorted
    * @param tile: percentile to compute (eg. 85 percentile)
    * @return value of input data at the specified percentile
    */
  def computePercentile(data: RDD[Double], tile: Double): Double = {
    // NIST method; data to be sorted in ascending order
    val r = data
    val c = r.count()
    if (c == 1) r.first()
    else {
      val n = (tile / 100d) * (c + 1d)
      val k = math.floor(n).toLong
      val d = n - k
      if (k <= 0) r.first()
      else {
        val index = r.zipWithIndex().map(_.swap)
        val last = c
        if (k >= c) {
          index.lookup(last - 1).head
        } else {
          index.lookup(k - 1).head + d * (index.lookup(k).head - index.lookup(k - 1).head)
        }
      }
    }
  }

  def filterByPercentile3(ds: Dataset[LeanItem3], bottom: Int, top: Int)(implicit sparkSession: SparkSession) = {
    val priced: RDD[Double] = ds.rdd.flatMap(_.price).sortBy(x => x)
    val topPerc = computePercentile(priced, top)
    val bottomPerc = computePercentile(priced, bottom)

    ds.filter(_.price.exists(x => x <= topPerc && x >= bottomPerc))
  }

  def normalizeText(ds: Dataset[LeanItem3])(implicit sparkSession: SparkSession) = {
    import sparkSession.implicits._
    val rInches = new Regex("(^| )(\\d+)(''|\")")
    val rMm = new Regex("(^| )(\\d+) ?mm($| )")
    val rCm = new Regex("(^| )(\\d+) ?cm($| )")
    val rCM = new Regex("(^| )(\\d+) ?m($| )")
    val rLts = new Regex("(^| )(\\d+) ?(lts|l)($| )")
    //    val rNumbers = new Regex("(^| )(\\d+)(''|\"|$)")

    ds.map(x => {
      val lowercase = x.title.toLowerCase
      val normalized = Normalizer.normalize(lowercase, Normalizer.Form.NFD).replaceAll("[^\\p{ASCII}]", "")
      val inches = rInches.replaceAllIn(normalized, "$2 pulgadas")
      val mm = rMm.replaceAllIn(inches, "$2 milimetros")
      val cm = rCm.replaceAllIn(mm, "$2 centimetros")
      val m = rCM.replaceAllIn(cm, "$2 metros")
      val lts = rLts.replaceAllIn(m, "$2 litros")
      //      val nums = rNumbers.replaceAllIn(lts, "[]")

      if(x.title != lts) x.copy(title = lts) else x
    })
  }
}
