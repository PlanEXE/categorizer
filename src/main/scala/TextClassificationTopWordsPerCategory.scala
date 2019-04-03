

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import java.text.Normalizer

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature.{CountVectorizer, NGram, RegexTokenizer, VectorAssembler, StringIndexer, IndexToString}
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import resource._

import scala.reflect.ClassTag
import scala.util.matching.Regex


//Naive bayes classfication + hashingTF
//Test set accuracy = 0.9507696051590923 => 1000000 features
//Test set accuracy = 0.9578104399628888 => 500000 features
//Test set accuracy = 0.9549639760654537 => 100000 features
//Test set accuracy = 0.9477678191392331 => 50000 features
//Test set accuracy = 0.8809176098332725 => 5000 features
//TODO: reduce features !

//Test set accuracy = 0.7897178702471406 => 1M Features - CAT_0
//Test set accuracy = 0.6884807772179786 - CAT_1 (Sin separar)

object TextClassificationTopWordsPerCategory extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

//        implicit val sparkSession = SparkSession.builder
//          .master("local[*]")
//          .appName("Spark Word Count")
//          .config("spark.driver.maxResultSize", "10g")
//          .getOrCreate()

    implicit val sparkSession = SparkSession.builder
      .appName("TextClassificationTopWordsPerCategory 02")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    import sparkSession.implicits._

//            val ds = normalizeText(sparkSession.read.json("./cellphones/*").as[LeanItem3]).as[LeanItem3]
    //    val ds = normalizeText(sparkSession.read.json("/home/ubuntu/SparkML/dataset/*").repartition(144*2).as[LeanItem3]).as[LeanItem3]
    sparkSession.sparkContext.hadoopConfiguration.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sparkSession.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", "AKIAJ5HUWSP2TUXBV3OA")
    sparkSession.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", "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")
    val ds = normalizeText(sparkSession.read.json("s3a://wolla.datasets/ml-dataset/dataset/*").repartition(552*3).as[LeanItem3]).as[LeanItem3]

    def savePipeline(name: String, sparkDataframe: DataFrame, transformers: Transformer*): Unit = {
      val transformedDataset = transformers.foldLeft(sparkDataframe)((x,y) => y.transform(x))
      val fullPath = s"jar:file:/home/ubuntu/SparkML/results/${name.trim}.zip"
      //new Pipeline().setStages(transformers.toArray).write.overwrite().save(s"/home/ubuntu/SparkML/results/${name.trim}-SPARK")
//      val fullPath = s"jar:file:/home/martin/dev/wolla/SparkML/results/${name.trim}.zip"
      println(fullPath)
      implicit val sbc: SparkBundleContext = SparkBundleContext().withDataset(transformedDataset)
      //            for(bf <- managed(BundleFile(s"jar:file:/home/martin/dev/wolla/SparkML/results/${name.trim}.zip"))) {
      for(bf <- managed(BundleFile(fullPath))) {
        SparkUtil.createPipelineModel(transformers.toArray).writeBundle.save(bf)(sbc).get
      }
    }

    //--------------PRE-PROCESS-------------

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

    def assembleVectors(cvs: Array[CountVectorizer]) = {
      new VectorAssembler()
        .setInputCols(cvs.map(_.getOutputCol).toArray)
        .setOutputCol("features")
    }

    def createVectorizers(key: String) = {
      val cv = new CountVectorizer()
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol(s"featuresWords_${key}")
        .setBinary(true)
        .setVocabSize(2000)
        //.setMinDF(50)

      key -> Array(cv)
    }


    val transformPipeline = new Pipeline().setStages(Array(tokenizer, ngram2, ngram3)).fit(ds)

    val transformedItems = transformPipeline.transform(ds).as[LeanItem3]

    val vectorizers = transformedItems.groupByKey(_.cat_0).keys.collect.map(createVectorizers)

    val vectorizersModels = vectorizers.map{ case(key, vecs) => new Pipeline().setStages(vecs).fit(transformedItems.filter(_.cat_0 == key)) }

    val va = assembleVectors(vectorizers.flatMap(_._2))

    val preProcessPipeline = new Pipeline().setStages(vectorizersModels :+ va).fit(transformedItems)

    val preProcessedItems = preProcessPipeline.transform(transformedItems).as[LeanItem3].cache()

    savePipeline("preProcess", ds.toDF(), transformPipeline, preProcessPipeline)

    //--------------PRE-PROCESS-------------

    //    val preProcessedItems = preProcessPipeline.transform(ds).as[LeanItem3]

    //preProcessedItems.show(10, false)


    def trainBayes(f: LeanItem3 => String, category_column: String): Unit = {

      val g = preProcessedItems.groupByKey(f).keys.collect()

      println(s"Groups: ${g.length} for: $category_column")

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

      val g = preProcessedItems.groupByKey(f).keys.collect()

      println(s"Groups: ${g.length} for: $category_column")

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
        val logRegModel = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").fit(labeledItems)

        savePipeline(s"$cat_n-$category_column", labeledItems.toDF(), logRegModel, converter)

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

    //        val naiveBayesModel = new LogisticRegression().setFamily("multinomial").setFeaturesCol("selectedFeatures").setLabelCol("label").fit(selector.transform(labeledItems))
    val naiveBayesModel = new NaiveBayes().setModelType("bernoulli").setFeaturesCol("features").setLabelCol("label").fit(labeledItems)

    //    val proc = converter.transform(naiveBayesModel.transform(labeledItems))

    savePipeline(s"cat_0", preProcessedItems.toDF(), naiveBayesModel, converter)
    println(s"cat_0")

//    trainBayes((x) => x.cat_0, "cat_1")
    trainLogReg(_.cat_0, "cat_1")
    trainLogReg(_.cat_1, "cat_2")

    sparkSession.stop()

    //    val Array(trainDF, testDF) = dsProcessed.randomSplit(Array(0.9, 0.1))

    //    val naiveBayesModel = new NaiveBayes().set: Dataset[LeanItem3] FeaturesCol("features").setLabelCol("label").fit(trainDF)

    //naiveBayesModel.transform(hashed).show()
    //    val predictions = naiveBayesModel.transform(testDF)
    //    //predictions.map{ case Row(t: String,l,c: String,w,f,rp,pr,p: Double) => (t, c, categoriesByIndex(p.toInt)(0)._1)}.show(100, false)
    //
    //
    //    // Select (prediction, true label) and compute test error
    //    val evaluator = new MulticlassClassificationEvaluator()
    //      .setLabelCol("label")
    //      .setPredictionCol("prediction")
    //      .setMetricName("accuracy")
    //    val accuracy = evaluator.evaluate(predictions)
    //    println("Test set accuracy = " + accuracy)

    //    println(s"Amount of categories in cat_0: ${ds.groupByKey(x => x.cat_0).keys.count()}")
    //    println(s"Amount of categories in cat_1: ${ds.groupByKey(x => x.cat_1).keys.count()}")
    //    println(s"Amount of categories in cat_1: ${ds.groupByKey(x => x.cat_2).keys.count()}")

    //    naiveBayesModel.save("./myNaiveBayesModel-1M-features")

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


  /**
    * compute percentile from an unsorted Spark RDD
    * @param data: input data set of Long integers
    * @param topTile: percentile to compute (eg. 85 percentile)
    * @return value of input data at the specified percentile
    */
  def filterPercentiles[T: ClassTag](data: RDD[T], filter: T => Boolean, field: T => Double, downTile: Double, topTile: Double): RDD[T] = {
    // NIST method; data to be sorted in ascending order
    val r = data.filter(filter).sortBy(field)
    val c = r.count()
    if (c == 1) data
    else {
      val nTop = (topTile / 100d) * (c + 1d)
      val kTop = math.floor(nTop).toLong

      val nDown = (downTile / 100d) * (c + 1d)
      val kDown = math.floor(nDown).toLong

      if (kTop <= 0 || kDown <= 0) data //TODO: CHECK OTHER POSIBILITIES
      else {
        val index: RDD[(Long, T)] = r.zipWithIndex().map(_.swap)
        index.filterByRange(kDown, kTop).map(_._2)
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
    val priced: RDD[Double] = ds.rdd.flatMap(_.price).sortBy(x => x)
    val topPerc = computePercentile(priced, top)
    val bottomPerc = computePercentile(priced, bottom)

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
    val rInches = "(?<=^|\\s)(\\d+)(''|\")($|\\s)"
    val rMm = "(?<=^|\\s)(\\d+) ?mm($| )"
    val rCm = "(?<=^|\\s)(\\d+) ?cm($| )"
    val rCM = "(?<=^|\\s)(\\d+) ?m($| )"
    val rLts = "(?<=^|\\s)(\\d+) ?(lts|l)($|\\s)"
    val rNumbers = "(?<=^|\\s)(\\d+|\\d+\\.\\d+|\\d+,\\d+)($|\\s|\\.$|\\.\\s|,\\s)"
    val rPunctuation = "[^\\w\\s\\+'\"/]"
    val rPunctuation2 = "\\s/\\s"
    val rSpaces = "\\s+"

    ds.map(x => {
      val newTitle = Normalizer.normalize(x.title.toLowerCase.trim, Normalizer.Form.NFD).replaceAll("[^\\p{ASCII}]", "")
        .replaceAll(rInches, "$1 pulgadas$3")
        .replaceAll(rMm, "$1 milimetros$2")
        .replaceAll(rCm, "$1 centimetros$2")
        .replaceAll(rCM, "$1 metros$2")
        .replaceAll(rLts, "$1 litros$3")
        .replaceAll(rNumbers, "NUMBER $2")
        .replaceAll(rPunctuation, " ")
        .replaceAll(rPunctuation2, " ")
        .replaceAll(rSpaces, " ")

      if(x.title != newTitle) x.copy(title = newTitle) else x
    })
  }
}
