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


//Naive bayes classfication + hashingTF
//Test set accuracy = 0.9507696051590923 => 1000000 features
//Test set accuracy = 0.9578104399628888 => 500000 features
//Test set accuracy = 0.9549639760654537 => 100000 features
//Test set accuracy = 0.9477678191392331 => 50000 features
//Test set accuracy = 0.8809176098332725 => 5000 features
//TODO: reduce features !

//Test set accuracy = 0.7897178702471406 => 1M Features - CAT_0
//Test set accuracy = 0.6884807772179786 - CAT_1 (Sin separar)

object TextClassificationTests2 extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

//    implicit val sparkSession = SparkSession.builder
//      .appName("Text Classification 2.0")
//      .config("spark.driver.maxResultSize", "10g")
//      .getOrCreate()

        implicit val sparkSession = SparkSession.builder
          .master("local[*]")
          .appName("Spark Word Count")
          .config("spark.driver.maxResultSize", "10g")
          .getOrCreate()

    val sc = sparkSession.sparkContext

    import sparkSession.implicits._

    val ds = normalizeText(sparkSession.read.json("./cellphones/*").repartition(4).as[LeanItem3])
//        sc.hadoopConfiguration.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
//        sc.hadoopConfiguration.set("fs.s3a.access.key", "AKIAJ5HUWSP2TUXBV3OA")
//        sc.hadoopConfiguration.set("fs.s3a.secret.key", "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")
//        val ds = normalizeText(sparkSession.read.json("s3a://wolla.datasets/ml-dataset/dataset/*").repartition(552*2).as[LeanItem3])

    def savePipeline(name: String, sparkDataframe: DataFrame, transformers: Transformer*): Unit = {
//      val transformedDataset = transformers.foldLeft(sparkDataframe)((x,y) => y.transform(x))
//      val fullPath = s"jar:file:/home/ubuntu/SparkML/results/${name.trim}.zip"
////            val fullPath = s"jar:file:/home/martin/dev/wolla/SparkML/results/${name.trim}.zip"
//      println(fullPath)
//      implicit val sbc: SparkBundleContext = SparkBundleContext().withDataset(transformedDataset)
//      //            for(bf <- managed(BundleFile(s"jar:file:/home/martin/dev/wolla/SparkML/results/${name.trim}.zip"))) {
//      for(bf <- managed(BundleFile(fullPath))) {
//        SparkUtil.createPipelineModel(transformers.toArray).writeBundle.save(bf)(sbc).get
//      }
    }

    //--------------PRE-PROCESS-------------

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    //    StopWordsRemover.loadDefaultStopWords("spanish")

    //    val remover = new StopWordsRemover()
    //      .setInputCol(tokenizer.getOutputCol)
    //      .setOutputCol("wordsRemoved")
    //      .setOutputCol("wordsFiltered")

    val ngram2 = new NGram()
      .setN(2)
      .setInputCol("words")
      .setOutputCol("2gram")

    val ngram3 = new NGram()
      .setN(3)
      .setInputCol("words")
      .setOutputCol("3gram")

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

    //    val cv =  new HashingTF()
    //      .setInputCol(remover.getOutputCol)
    //      .setOutputCol("features")
    //      .setNumFeatures(100000)


    val columnDropper = new SQLTransformer().setStatement("SELECT title, cat_0, cat_1, cat_2, price, features FROM __THIS__")

    val preProcessPipeline = new Pipeline().setStages(Array(tokenizer, ngram2, ngram3, cv, cvGram2, cvGram3, va)).fit(ds)

    val preProcessedItems = columnDropper.transform(preProcessPipeline.transform(ds)).as[LeanItem3].cache()

    preProcessedItems.show(10, false)

    savePipeline("preProcess", ds.toDF(), preProcessPipeline)

    //--------------PRE-PROCESS-------------

    //    val preProcessedItems = preProcessPipeline.transform(ds).as[LeanItem3]

    //preProcessedItems.show(10, false)


    def trainBayes(f: LeanItem3 => String, category_column: String): Unit = {

      val g = preProcessedItems.groupByKey(f).keys.collect().par

      println(s"Groups: ${g.length} for: $category_column")

      g.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(2))
      g.foreach(cat_n => {
        val itemDs = filterByPercentile3(preProcessedItems.filter((x) => f(x) == cat_n), 5, 95)

        val labelIndexer = new StringIndexer()
          .setInputCol(category_column)
          .setOutputCol("label")
          .fit(itemDs)

        val converter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("categories")
          .setLabels(labelIndexer.labels)

        val labeledItems = labelIndexer.transform(itemDs).as[LeanItem3]

        //        val naiveBayesModel = new RandomForestClassifier().setMaxDepth(30).setNumTrees(10).setFeaturesCol("features").setLabelCol("label").fit(labeledItems)
        //                val naiveBayesModel = new LogisticRegression().setFamily("multinomial").setFeaturesCol("features").setLabelCol("label").fit(labeledItems)
        val naiveBayesModel = new NaiveBayes().setModelType("bernoulli").setFeaturesCol("features").setLabelCol("label").fit(labeledItems)

        savePipeline(s"$cat_n-$category_column", labeledItems.toDF(), naiveBayesModel, converter)

        println(s"$cat_n-$category_column")
      })
    }

    def trainLogReg(f: LeanItem3 => String, category_column: String): Unit = {

      val g = preProcessedItems.groupByKey(f).keys.collect().par

      println(s"Groups: ${g.length} for: $category_column")

      g.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(2))
      g.foreach(cat_n => {
        val itemDs = filterByPercentile3(preProcessedItems.filter((x) => f(x) == cat_n), 5, 95).cache()

        val labelIndexer = new StringIndexer()
          .setInputCol(category_column)
          .setOutputCol("label")
          .fit(itemDs)

        val converter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("categories")
          .setLabels(labelIndexer.labels)

        val labeledItems = labelIndexer.transform(itemDs).as[LeanItem3]

        //        val naiveBayesModel = new RandomForestClassifier().setMaxDepth(30).setNumTrees(10).setFeaturesCol("features").setLabelCol("label").fit(labeledItems)
        val logRegModel = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").fit(labeledItems)

        testAccuracy(logRegModel, itemDs, converter)

        itemDs.unpersist()
        savePipeline(s"$cat_n-$category_column", labeledItems.toDF(), logRegModel, converter)

        println(s"$cat_n-$category_column")
      })
    }

    def trainNN(f: LeanItem3 => String, category_column: String): Unit = {

      val g = preProcessedItems.groupByKey(f).keys.collect().par

      println(s"Groups: ${g.length} for: $category_column")

      g.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(1))
      g.foreach(cat_n => {
        val itemDs = filterByPercentile3(preProcessedItems.filter((x) => f(x) == cat_n), 5, 95).cache()

        val labelIndexer = new StringIndexer()
          .setInputCol(category_column)
          .setOutputCol("label")
          .fit(itemDs)

        val converter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("categories")
          .setLabels(labelIndexer.labels)

        val selector = new ChiSqSelector().setNumTopFeatures(50).setLabelCol("label").setFeaturesCol("features").fit(itemDs)

        val labeledItems = labelIndexer.transform(itemDs).as[LeanItem3]

        val inputLayer = 50//vcs.map(_.getVocabSize).sum
        val model = new MultilayerPerceptronClassifier().setBlockSize(256).setLayers(Array(inputLayer, inputLayer/2, labelIndexer.labels.length)).setFeaturesCol("featuresSelected").setLabelCol("label").fit(labeledItems)

        itemDs.unpersist()
        savePipeline(s"$cat_n-$category_column", labeledItems.toDF(), selector, model, converter)

        println(s"$cat_n-$category_column")
      })
    }

    val labelIndexer = new StringIndexer()
      .setInputCol("cat_0")
      .setOutputCol("label")
      .fit(preProcessedItems)

    val converter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("categories")
      .setLabels(labelIndexer.labels)

    //    savePipeline(s"converter-cat_0", converter)

    val labeledItems = labelIndexer.transform(preProcessedItems).as[LeanItem3]

    //        val naiveBayesModel = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").fit(labeledItems)
    val naiveBayesModel = new NaiveBayes().setModelType("bernoulli").setFeaturesCol("features").setLabelCol("label").fit(labeledItems)
    //    val proc = converter.transform(naiveBayesModel.transform(labeledItems))

    savePipeline(s"cat_0", preProcessedItems.toDF(), naiveBayesModel, converter)
    println(s"cat_0")

    trainBayes(_.cat_0, "cat_1")
    trainLogReg(_.cat_1, "cat_2")
    //    trainLogReg(_.cat_1, "cat_2")

    println("FINISHED!!!")
    sparkSession.stop()

  }

  def testAccuracy(model: LogisticRegressionModel, dsProcessed: Dataset[LeanItem3], converter: IndexToString)(implicit sparkSession: SparkSession) = {
    import sparkSession.implicits._

    val keys = dsProcessed.groupByKey(_.cat_2).keys.collect()

    keys.foreach{ g =>
      val testDF  = dsProcessed.filter(_.cat_2 == g)
      val predictions = model.transform(testDF)

      converter.transform(predictions).show(10, false)

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println(s"Test set accuracy for cat_2: [$g] = $accuracy")
    }
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


  def filterByPercentile(ds: Dataset[LeanItem3])(implicit sparkSession: SparkSession) = {
    import sparkSession.implicits._

    ds.groupByKey(_.cat_2).keys.collect().par.map(g => {
      val filtered = ds.filter(_.cat_2 == g)
      val priced: RDD[Double] = filtered.rdd.flatMap(_.price)
      val p85 = computePercentile(priced, 90)
      val p15 = computePercentile(priced, 10)
      println(g)
      println(p15)
      println(p85)

      filtered.filter(_.price.exists(x => x <= p85 && x >= p15))
    }).reduce((x,y) => x.union(y))
  }

  def filterByPercentile3(ds: Dataset[LeanItem3], bottom: Int, top: Int)(implicit sparkSession: SparkSession) = {
    val priced: RDD[Double] = ds.rdd.flatMap(_.price).sortBy(x => x).cache()
    val topPerc = computePercentile(priced, top)
    val bottomPerc = computePercentile(priced, bottom)
    priced.unpersist()

    ds.filter(_.price.exists(x => x <= topPerc && x >= bottomPerc))
  }

  //  def filterByPercentile2(ds: Dataset[LeanItem3])(implicit sparkSession: SparkSession) = {
  //    import sparkSession.implicits._
  //
  //    ds.groupByKey(_.cat_2).keys.collect().par.map(g => {
  //      val filtered = ds.filter(_.cat_2 == g)
  //      filterPercentiles[LeanItem3](filtered.rdd, x => x.price.getOrElse(0), 10, 90)
  //    }).reduce((x,y) => x.union(y))
  //  }

  //  def filterOutilers(ds: Dataset[LeanItem3])(implicit sparkSession: SparkSession) = {
  //    import sparkSession.implicits._
  //    ds.filter(x => x.price.exists(p => p > 1 && p < 999999))
  //  }

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

      if(x.title != newTitle) x.copy(title = newTitle) else x
    })
  }
}
