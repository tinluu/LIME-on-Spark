'''
Import all of the following.

Note that we do not work and importing Pandas DataFrame as Spark ML uses its own DF structure that is RDD
In addition, converting between PD DF and Spark DF is a waste of efficiency 
'''
import lime 
from lime import lime_text
from lime.lime_text import LimeTextExplainer

import numpy as np

import pyspark
from pyspark import SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql import functions as F

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics


sc = SparkContext()

spark = SparkSession \
        .builder \
        .appName("Lime-in-Spark") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

########################################################################################################
'''
Load dataset 20 Newsgroups
This publicly available data has been uploaded to my HFS (/user/tbl245/) 
'''
categories = ["soc.religion.christian", "alt.atheism"]
LabeledDocument = pyspark.sql.Row("category", "text")

def categoryFromPath(path):
    return path.split("/")[-2]
    
def prepareDF(typ):
    rdds = [sc.wholeTextFiles("/user/tbl245/20news-bydate-" + typ + "/" + category)\
              .map(lambda x: LabeledDocument(categoryFromPath(x[0]), x[1]))\
            for category in categories]
    return sc.union(rdds).toDF()

train_df = prepareDF("train").cache()
test_df  = prepareDF("test").cache()

# train_df.count()
# number of documents in training set: 1079
# test_df.count()
# number of documents in test set: 717

'''
Index all documents so that we can select any particular documents we wish.
There are many efficient ways to create an 0-based index column, below is one way.
'''
def get_index_column(df):
    df_final = df.rdd.zipWithIndex().toDF()
    df_final = df_final.withColumn('category', df_final['_1'].getItem("category"))
    df_final = df_final.withColumn('text', df_final['_1'].getItem("text"))
    df_final = df_final.withColumn('id', df_final['_2'])
    df_final = df_final.drop("_1")
    df_final = df_final.drop("_2")
    return df_final

train_df = get_index_column(train_df)
test_df = get_index_column(test_df)

########################################################################################################
'''
Below, we build an ML pipeline to classify the sentiment of these documents.
Pipeline transformers: Indexer, Tokenizer, TF-idf
Classifier: Logistic regression with L2 regularization (0.001)

A couple of IMPORTANT notes regarding the pipeline:
1) The choice of word tokenizer for the pipeline is deliberate and important. We want a tokenizer that, first, preserves as much sentiment signal from the words as possible (capitalized "God" has different sentiment from lowercase "god"). Secondly, our tokenzier should parse documents exactly the same way as how LIME does it (pattern=u'\W+'). Inconsistent parsing will lead to inconsistent accuracy and textual explanation. Therefore, we choose the flexible regular-expression tokenizer (RegexTokenizer).
2) Performing TF-idf requires two steps, hashing TF and idf. 
3) StringIndexer will binarize the two categories such as "Christianity" is labeled as 0, "Atheism" as 1.
'''
indexer   = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="text", outputCol="all_words", toLowercase=False)
hashingTF = HashingTF(inputCol="all_words", outputCol="rawFeatures")
idf       = IDF(inputCol="rawFeatures", outputCol="features")
lr        = LogisticRegression(maxIter=20, regParam=0.001)

pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr])
model = pipeline.fit(train_df)
prediction = model.transform(test_df).cache()

'''
There are many columns in our Prediction dataframe that are the outputs of transformers in the pipeline.

Here are the (schema) column names and types:
DataFrame[category: string, text: string, id: bigint, label: double, all_words: array<string>, rawFeatures: vector, features: vector, rawPrediction: vector, probability: vector, prediction: double]

Not all of these columns are important. To survey this dataframe, we can filter out various columns we care about.
These columns can be: id, category, label, prediction, probability.

Example output: 
>>>> prediction.select("id", "category", "label", "prediction", "probability").show(truncate=False)
+---+----------------------+-----+----------+------------------------------------------+
|id |category              |label|prediction|probability                               |
+---+----------------------+-----+----------+------------------------------------------+
|0  |soc.religion.christian|0.0  |0.0       |[0.9995107515101412,4.892484898589173E-4] |
|1  |soc.religion.christian|0.0  |0.0       |[0.982377714690385,0.017622285309614957]  |
|2  |soc.religion.christian|0.0  |0.0       |[0.9999999260589753,7.394102482345168E-8] |
|3  |soc.religion.christian|0.0  |0.0       |[0.9983875117536608,0.0016124882463392641]|
|4  |soc.religion.christian|0.0  |0.0       |[0.9989878551234942,0.0010121448765058304]|
|5  |soc.religion.christian|0.0  |0.0       |[0.9984832811079428,0.0015167188920571287]|
+---+----------------------+-----+----------+------------------------------------------+

'''
metrics = MulticlassMetrics(prediction.select("label", "prediction").rdd)
f1_score = metrics.fMeasure()

'''
f1_score = 0.9483960948396095 (given the 717 documents in test set, 37 are misclassified)
Given this score as the baseline, we will use LIME and its textual explanations to improve this score.
'''
#####################################################################################################
'''
Using LIME, we will then remove chosen words from the training set and retrain our pipeline. Below,
we create a function that takes in the list of words to be removed from the training set, and output the 
new f1 score for the test set. 
'''
    # Remove these selected words from the training data set (not the test set!)
    # Use the same pipeline used to build baseline model and train on new training set to get new model
    # note that we make NO changes to test set
    
def train_eval_model(words, pipeline = pipeline, train_df = train_df, test_df = test_df):
    subtract_words_udf = F.udf(lambda text: ' '.join([word for word in text.split() if word not in words]), StringType())
    new_train_df =  train_df.withColumn("text",subtract_words_udf(F.col("text")))
    model = pipeline.fit(new_train_df)
    prediction = model.transform(test_df)
    metrics = MulticlassMetrics(prediction.select("label", "prediction").rdd)
    f1 = metrics.fMeasure() 
    return f1

########################################################################################################
'''
LIME 
To generate textual explanations for output predictions of various documents, ideally, this is how LIME
would be executed in scikit-learn:

    from sklearn.pipeline import make_pipeline
    c = make_pipeline(vectorizer, sdg)
    explainer = LimeTextExplainer(class_names = labels)
    exp = explainer.explain_instance(doc_text, c.predict_proba, num_features=6)

Note that LIME explainer directly uses predict_prob (gives probability for each class) from sklearn.pipeline.
Since we are integrating Spark with LIME and the former is not native to the latter, and also that Spark ML
does not have a function that gives just the probability prediction, we have to build this function. This 
turns out to be an advantage (to have to build this function) because we can make this function, which is 
required by LIME, native to Spark (taking advantage of Spark data parallelization and structure). 

'''
labels = ['Christianity', 'Atheism']
explainer = LimeTextExplainer(class_names = labels)
 
def classifier_fn(data):
    spark_object = spark.createDataFrame(data, "string").toDF("text")
    pred = model.transform(spark_object) 
    output = np.array((pred.select("probability").collect())).reshape(len(data),2)
    return output

'''
We can now generate LIME textual explanations with:
exp = explainer.explain_instance(doc_text, classifier_fn, num_features=6)

num_features is the number of words that would be used as textual explanations for the prediction. This
parameter can be chosen by us. In our cause, we choose have 6 words as textuals explanations. 

'''
# Below, we generate prediction outputs and LIME textual explanations for 3 chosen documents in the test set.
# Chosen document ID's: 0, 163, 401

for doc_id in [0, 163, 401]:
    
    print("\nTest set document id: %s" % doc_id)
    doc = prediction.filter(prediction.id == doc_id)
    
    doc_cat = doc.select("label").collect()[0][0]
    doc_pred = doc.select("prediction").collect()[0][0]
    doc_probs = doc.select("probability").collect()[0][0]
    
    doc_text = doc.select("text").collect()[0][0]
    lime = explainer.explain_instance(doc_text, classifier_fn, num_features=6).as_list()
    
    print('True class: %s' % labels[int(doc_cat)])
    print('Predicted class: %s' % labels[int(doc_pred)])
    print('Prob(Christianity) = %s |' % doc_probs[0], 'Prob(Atheism) = %s' % doc_probs[1])
    print('LIME explanation', lime)

'''
The output for the above documents is as follow:

Test set document id: 0
True class: Christianity
Predicted class: Christianity
Prob(Christianity) = 0.9997855684733377 | Prob(Atheism) = 0.00021443152666238735
LIME explanation [('mail', -0.01910505600702484), ('proponents', -0.018191805700043362), ('Moderator', -0.015837249507868065), ('1993', -0.015617204336208924), ('ucsu', -0.0147896050825634), ('Organization', -0.01411873736300116)]

Test set document id: 163
True class: Christianity
Predicted class: Atheism
Prob(Christianity) = 0.026481900174021293 | Prob(Atheism) = 0.9735180998259786
LIME explanation [('mangoe', 0.07209706428745866), ('marv', 0.06310736595398016), ('Wingate', 0.06049618357773301), ('tove', 0.05920332485557272), ('lous', 0.05850466642304861), ('sod', 0.05198236049959123)]

Test set document id: 401
True class: Atheism
Predicted class: Christianity
Prob(Christianity) = 0.7979931161746718 | Prob(Atheism) = 0.20200688382532808
LIME explanation [('Freemasonry', -0.23266657989771578), ('Page', -0.10040317339115276), ('equality', 0.091947667371753), ('10th', -0.07641395045259208), ('Ministry', 0.07013575249343644), ('NNTP', 0.06018550355787116)]
'''
############################################################################################################
'''
The difference between the imputed probabilities of the two classes for a document, denoted "conf", can be a measure of the model’s confidence in its prediction for that document.  
 
     conf = |prob(Christian)-prob(Atheism)| 

Thus, we generate explanations for all misclassified documents in the test set, sorted by "conf" in descending order
'''
# Get "conf" and order in descending manner

misclassified = prediction.filter(prediction.label != prediction.prediction)
subtract_vector_udf = F.udf(lambda arr: float(np.abs(arr[0]-arr[1])), DoubleType())
conf = misclassified.withColumn("conf", subtract_vector_udf(F.col('probability')))
conf_ordered = conf.select("id", "label", "text", "probability", "conf").orderBy(conf.conf, ascending = False).cache()

# Generate textual explanations for each misclassified documents

misclassified_texts = conf_ordered.select("id", "text").collect()
lime_explanations = []

for text in misclassified_texts:
    textual_exp = explainer.explain_instance(text[1], classifier_fn, num_features = 6).as_list()
    lime_explanations.append((text[0], textual_exp))
    
lime_df = [(i[0], "[%s, %s, %s, %s, %s, %s]" % tuple(i[1])) for i in lime_explanations]
lime_df = sc.parallelize(lime_df).toDF(["id", "explanations"])
misclassified_lime = conf_ordered.join(lime_df, conf_ordered.id == lime_df.id).select(conf_ordered.id, "conf", "explanations")

'''
misclassified_lime looks like this:
+---+------------------+--------------------+
| id|              conf|        explanations|
+---+------------------+--------------------+
|682|0.9999996074475812|[('resurection', ...|
|666|0.9501253240501562|[('ctron', -0.480...|
|715|0.9499227792301793|[('Conversions', ...|
|163|0.9470361996519594|[('mangoe', 0.074...|
|704|0.9313821629681229|[('formerly', -0....|
|424|0.9289098235457514|[('Astrophysics',...|
|512|0.9097972691419043|[('Alan', -0.0877...|
|588|0.8873918932180136|[('ocis', -0.1938...|
|523| 0.877367361283441|[('influenced', -...|
|610|0.8713297232624089|[('ocis', -0.1961...|
+---+------------------+--------------------+
'''
#######################################################################################################
'''
For each of the misclassified documents in the test set, we have a list of LIME generated textual explanations - words that contributed to the misclassfication of the document. The goal of LIME is to identify some of the resilient words from LIME explanations that we can use to remove from the training set to improve the performance of our model. 

Note: there is no causal relationship between the generated explanatory features and the predictions. 
'''

words = {}

for key, value in dict(lime_explanations).items():
    for word_weight in value:
        if word_weight[0] not in words:
            words[word_weight[0]] = [np.abs(word_weight[1]), 1]
        else:
            value = words[word_weight[0]]
            words[word_weight[0]] = [value[0] + np.abs(word_weight[1]), value[1] + 1]

temp = []
for key, value in words.items():
    temp.append([str(key), float(value[0]), int(value[1])])
words_weights_counts = sc.parallelize(temp).toDF(["word", "weight", "count"])

'''
Words (features) ordered by "count" or how many misclassified documents they have contributed to:
>>> words_weights_counts.orderBy("count", ascending = False).show()
+------------+-------------------+-----+
|        word|             weight|count|
+------------+-------------------+-----+
|Organization| 0.7842242569798894|   14|
|  Newsreader| 0.5258396945216637|    7|
|        Host| 0.5063339338367208|    7|
|     Posting| 0.5315062361996961|    7|
|        Carl| 0.8223884072724564|    5|
|     sincere| 0.3126740588463568|    4|
|         fsu| 0.1183629639500737|    3|
|       astro| 0.5759097192031847|    3|
|        ocis|  0.584776543472326|    3|
|        Alan|0.39177755717677554|    3|
|      Asimov| 0.3306991216581148|    3|
|       stamp|0.13211372928470771|    2|
+------------+-------------------+-----+

Words (features) ordered by cummulative "weight" or how much they contributed misclassified documents:
>>> words_weights_counts.orderBy("weight", ascending = False).show()
+------------+-------------------+-----+
|        word|             weight|count|
+------------+-------------------+-----+
|        Carl| 0.8223884072724564|    5|
|Organization| 0.7842242569798894|   14|
|        ocis|  0.584776543472326|    3|
|       astro| 0.5759097192031847|    3|
|     Posting| 0.5315062361996961|    7|
|  Newsreader| 0.5258396945216637|    7|
|        Host| 0.5063339338367208|    7|
|       ctron| 0.4803020898300515|    1|
|        Alan|0.39177755717677554|    3|
|      Asimov| 0.3306991216581148|    3|
|     sincere| 0.3126740588463568|    4|
| Tribulation|  0.286633907578331|    1|
+------------+-------------------+-----+
'''

### STRATEGY 1:
'''
There are many strategies to choose words to remove, in our case, we choose features who weights are at least 0.1
Below is the generated set of words to remove:

words_to_remove = ['loud', 'ctron', 'Conversions', 'fsu', 'Organization', 'Astrophysics', 'Host', 'Alan', 'voices', 'ocis', 'astro', 'Carl', 'stamp', 'Newsreader', 'Posting', 'survey', 'sincere', 'warns', 'refuses', 'sinners', 'beast', 'Tribulation', 'Antichrist', 'Freemasonry', 'Page', 'Asimov', 'constructive', 'Todd', 'LaTech', 'Supercomputer', 'Computations', 'Caused', 'agonizing', 'carpenter', 'Path', 'output', 'Arlington', 'feedback', 'Julian', 'consumers', 'Rabbi', 'scriptures', 'Return', 'Scouts', 'Boy', 'creationists', 'math', 'hadith', 'expertise', 'discussion', 'Qur', 'Knowledge', 'Quoting']

'''
words_to_remove_1 = words_weights_counts.filter(words_weights_counts.weight >= 0.1).select("word")
words_to_remove_1 = [i[0] for i in words_to_remove.collect()]
train_eval_model(words_to_remove_1) 
'''
STRATEGY 1 leads to the new F1-score = 0.9539748953974896 (versus the benchmark f1_score = 0.9483960948396095)
'''

### STRATEGY 2:
words_to_remove_2 = ['temple','we', 'Christ', 'the', 'and', 'of', 'to', 'God', 'love', 'Steve', 'uxa', 'morality', 'Christians', 'Christian', 'beliefs', 'wustl', 'Paul', 'faith', 'church']
train_eval_model(words_to_remove_2) 
'''
STRATEGY 2 leads to F1-score = 0.9595536959553695 (versus the benchmark f1_score = 0.9483960948396095)
'''

### STRATEGY 3:
'''
Consider misclassified documents whose confidence is at least 0.1 (higher certainty of incorrectness)
If label is Atheism, considers only negative-valued features. Opposite for Christianity. 
'''

conf_ordered_2 = conf_ordered.filter(conf_ordered.conf >= 0.1)

def find_mis_word(data):
    temp = data
    if temp[0] == 1.0:
        temp[1] = [x for x in temp[1] if x[1] < 0]
    else:
        temp[1] = [x for x in temp[1] if x[1] > 0]
    return temp

# Define a new explainer 
new_explainer = LimeTextExplainer(feature_selection='none', class_names = labels)

misclassified_texts_2 = conf_ordered_2.select("label","text").collect()
lime_explanations_2 = []

for text in misclassified_texts_2:
    textual_exp = new_explainer.explain_instance(text[1], classifier_fn, num_features = 6).as_list()
    lime_explanations_2.append([text[0], textual_exp])

# We select features whose cummulative weights are larger than 0.02
label_feature = sc.parallelize(lime_explanations_2).map(lambda x: find_mis_word(x)).flatMap(lambda x: x[1])\
                .filter(lambda x: abs(x[1]) > 0.02).map(lambda x: (x[0], (1, abs(x[1]))))\
                    .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))\
                        .map(lambda x: (str(x[0]),x[1][0],x[1][1])).\
                            sortBy(lambda x: x[1], ascending=False)
test_feature_df = label_feature.toDF(["word_j","count_j","weight_j"])

# We pick the top 100 features by weights
words_to_remove_3 = test_feature_df.orderBy(test_feature_df.weight_j, ascending=False).select("word_j").limit(100).rdd.flatMap(lambda x: x).collect()
'''
These words are: ['Organization', 'Carl', 'ocis', 'astro', 'ctron', 'Alan', 'sincere', 'Antichrist', 'agonizing', 'Return', 'Path', 'Freemasonry', 'Conversions', 'carpenter', 'beast', 'warns', 'Julian', 'Rabbi', 'feedback', 'math', 'sinners', 'constructive', 'fsu', 'scriptures', 'Astrophysics', 'Todd', 'loud', 'Page', 'voices', 'survey', 'Supercomputer', 'Tourist', 'Computations', 'KSAND', 'newton', 'Warning', 'Daniel', 'C6697n', 'deceived', 'Comp', 'Russell', 'indiana', 'disguise', 'smug', '10th', 'omnipresent', 'hole', 'wustl', 'mangoe', 'attitude', 'Florida', 'Bertrand', 'orst', 'uxa', 'Oser', 'fermi', 'cycle', 'influenced', 'lous', 'oser', 'Wingate', 'tove', 'marv', 'research', 'fill', 'Schnitzius', 'replies', 'fashioned', 'sod', 'cattle', 'atone', 'thank', 'Peter', 'formerly', 'Me', 'Romans', 'Love', 'Steve', 'Waco', 'faced', 'spat', 'jews', 'meet', 'deeds', 'conjunction', 'romans', 'NAME', 'verify', 'request', 'nonrepeatability', 'Stanley', 'Business', 'Thanks', 'through', 'Adelaide', 'William', 'indoctrination', 'interview', 'York', 'straws']
'''
train_eval_model(words_to_remove_3) 
'''
STRATEGY 3 leads to F1-score = 0.9623430962343096.
'''

'''
As you can see, we can improve the classifier accuracy by using LIME.
'''

