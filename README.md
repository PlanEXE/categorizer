# SparkML

The main proyect and the one used on production enviroments is TextClassification.
The other proyects are tests and proof of concept.

### Running mode

You can choose if you want to run in local mode o cluster mode:

Cluster Mode

        implicit val sparkSession = SparkSession.builder
          .appName("Text Classification 3.0 - Cluster Mode")
          .config("spark.driver.maxResultSize", "10g")
          .getOrCreate()

Local Mode using all cores for the machine

        implicit val sparkSession = SparkSession.builder
            .master("local[*]")
            .appName("Text Classification 3.0 - Cluster Mode")
            .config("spark.driver.maxResultSize", "10g")
            .getOrCreate()


The dataset can be read localy or remotely

Localy

    val ds = normalizeText(sparkSession.read.json("RUTA LOCAL/*").repartition(4*2).as[LeanItem3])

Remote (from s3)

    sc.hadoopConfiguration.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sc.hadoopConfiguration.set("fs.s3a.access.key", "******************")
    sc.hadoopConfiguration.set("fs.s3a.secret.key", "******************")
    val ds = normalizeText(sparkSession.read.json("s3a://wolla.datasets/ml-dataset/dataset/*").repartition(552*3).as[LeanItem3])

### normalizeText
normalizeText applies a series of regex rules in order to clean the dataset.
This rules include:
* ', '', " converted to pulgadas
* mm, milimetro, milimetros converted to milimetros
* cm, centimetros, centimetro converted to centimetros
* m, metro, metros converted to metros
* lts, l, litros, litro converted to litros
* ml, mililitros, mililitro converted to mililitros
* gr, grs, gramos, gramo converted to gramos, g is not used as 4g would not mean grams but cellphones
* kilo, kilos converted to kilogramos, k is not used as 4k would not mean kilos but tvs
* w, watt, watts converted to watts
* v, volt, volts converted to volts
* 20/30 converted to NUMBER / NUMBER
* 20x30 converted to NUMBER x NUMBER
* Numbers are transformed to NUMBER. ex: 12 12,3 12.3
* Consecutive numbers speraeted by commas are converted to NUMBER. ex: 1,2,3 -> NUMBER,NUMBER,NUMBER
* & is converted to y
* % is converted to porcentaje
* All punctuation symbols are removed except +
* All extra spaces are removed

This rules shuould be applied on the predictor on execution time as well as this is part of the pre process pipeline

Extra regex are applied in normalize text. This extra regex are not necesary on the predictor.

* "envio gratis" is removed from the text of every item.
* "local a la calle" is removed from the text of every item.
* "factura a o b" is removed from the text of every item.
* "factura a" is removed from the text of every item.
* "factura b" is removed from the text of every item.

### Pre-process pipeline

The preprocess piepline is composed of: tokenizer, remover, ngram2, ngram3, cv, cvGram2, cvGram3, va

* tokenizer: `Converts title in array of words, this output is returned in another column: words`
* remover: `Removes stopwords, this output is returned in another column: wordsFiltered`
* ngram2/ngram3: `Generates ngrams from the words column (uses the stopwords, ex: celular con funda), this output is returned in another column: ngrams2/ngrams3`
* cv/cvGram2/cvGram3: `CountVectorizers: Generated vector from detected words, this has to be trained with the dataset to know wich will be the words in existance. The max amount of words can be configured as well as the minimum amout of times each words must appear on the datset`
* va: `Vector assmbler: assembles the output off all the count vectorizers as one unique vector`

### Model training

We might use `Naive Bayes` with bernoulli or `Logistic Regression`.
Logistic regression is way more heavy on computations, so be aware of that.
Previous to the usage of any of this algorthims the dataset should have a `label` column used as the right answer.
This will use the `labelIndexer` wich converts every category to a number or "index".
At the same time we will need an `IndexToString` wich makes the reverse process. The `IndexToString` is necesary so we have our category back insted of a number that means nothing to us.

Using a `Neural Network` also yields great results but is more costly to train that `Logistic Regression`. The code is prepared to be used but has never been tested.