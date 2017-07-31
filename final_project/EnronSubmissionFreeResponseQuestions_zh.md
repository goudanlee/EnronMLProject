#优达学城数据分析师纳米学位项目 P5
##安然提交开放式问题

 说明：你可以在这里下载此文档的英文版本。

 机器学习的一个重要部分就是明确你的分析过程，并有效地传达给他人。下面的问题将帮助我们理解你的决策过程及为你的项目提供反馈。请回答每个问题；每个问题的答案长度应为大概 1 到 2 段文字。如果你发现自己的答案过长，请看看是否可加以精简！

 当评估员审查你的回答时，他或她将使用特定标准项清单来评估你的答案。下面是该标准的链接：评估准则。每个问题有一或多个关联的特定标准项，因此在提交答案前，请先查阅标准的相应部分。如果你的回答未满足所有标准点的期望，你将需要修改和重新提交项目。确保你的回答有足够的详细信息，使评估员能够理解你在进行数据分析时采取的每个步骤和思考过程。

 提交回答后，你的导师将查看并对你的一个或多个答案提出几个更有针对性的后续问题。

 我们期待看到你的项目成果！

1.向我们总结此项目的目标以及机器学习对于实现此目标有何帮助。作为答案的部分，提供一些数据集背景信息以及这些信息如何用于回答项目问题。你在获得数据时它们是否包含任何异常值，你是如何进行处理的？【相关标准项：“数据探索”，“异常值调查”】

* 此项目目标是根据公开的安然财务和电子邮件数据集，构建机器学习算法，找出有欺诈嫌疑的安然员工。

* 此数据集包含信息如下：
数据集包含146个数据样本，其中每个数据样本包含变量21个。

* 排除一个'Total'数据样本，剩余145个数据样本对应每一个安然员工，其中POI嫌疑人18人，非POI有127人。

* 每个数据样本的21个变量中，其中有20个特征（14个财务特征，6个邮件特征），1个嫌疑人标签。

* 缺失值情况：20个特征均有不同程度的缺失情况，即值为“NaN”，可以看到部分特征的缺失比例高于50%，最高的缺失高达98%，这里我们保留所有特征，后续通过SelectKBest来查看每个特征的score。

``` python

The ratio of the null values of each column:
[('loan_advances', 0.98),
 ('director_fees', 0.89),
 ('restricted_stock_deferred', 0.88),
 ('deferral_payments', 0.74),
 ('deferred_income', 0.67),
 ('long_term_incentive', 0.55),
 ('bonus', 0.44),
 ('to_messages', 0.41),
 ('from_poi_to_this_person', 0.41),
 ('from_messages', 0.41),
 ('from_this_person_to_poi', 0.41),
 ('shared_receipt_with_poi', 0.41),
 ('other', 0.37),
 ('salary', 0.35),
 ('expenses', 0.35),
 ('exercised_stock_options', 0.3),
 ('restricted_stock', 0.25),
 ('email_address', 0.23),
 ('total_payments', 0.14),
 ('total_stock_value', 0.14)]

```


* 异常值处理：
数据集中包含一条‘Total’数据，是其他数据的汇总，我们将其视为异常值，并进行了移除。

对该数据的分析我们可以了解到以下信息，帮助我们选择后续的机器学习策略：

* 这个数据集很不平衡（imbalance）, 也就说明accuracy并不是很好的评估指标，选择precision和recall更好一些。

* 在交叉验证的时候，因为数据的不平衡性，我们会选用Stratified Shuffle Split的方式将数据分为验证集和测试集。

* 数据样本比较少，因此我们可以使用GridSearchCV来进行参数调整，如果较大的数据则会花费较长的时间，可以考虑使用RandomizedSearchCV.


2.你最终在你的 POI 标识符中使用了什么特征，你使用了什么筛选过程来挑选它们？你是否需要进行任何缩放？为什么？作为任务的一部分，你应该尝试设计自己的特征，而非使用数据集中现成的——解释你尝试创建的特征及其基本原理。（你不一定要在最后的分析中使用它，而只设计并测试它）。在你的特征选择步骤，如果你使用了算法（如决策树），请也给出所使用特征的特征重要性；如果你使用了自动特征选择函数（如 SelectBest），请报告特征得分及你所选的参数值的原因。【相关标准项：“创建新特征”、“适当缩放特征”、“智能选择功能”】

* 最终使用的特征采用SelectKBest进行了筛选，最终的模型使用了6个最佳特征：
['salary', 'bonus', 'total_stock_value', 'exercised\_stock\_options', 'to\_poi\_ratio', 'deferred\_income']

使用SelectKBest筛选特征时，特征的重要性得分如下:

可以看到我们创建的新特征“to\_poi\_ratio”获得了较好的重要性得分，意味着该特征对于我们寻找POI是有帮助的，而另外一个新特征“from\_poi\_ratio”则表现不佳，其Score值仅为3.21。

``` python

[('exercised_stock_options', 25.097541528735491),
 ('total_stock_value', 24.467654047526398),
 ('bonus', 21.060001707536571),
 ('salary', 18.575703268041785),
 ('to_poi_ratio', 16.641679564325358),
 ('deferred_income', 11.595547659730601),
 ('long_term_incentive', 10.072454529369441),
 ('restricted_stock', 9.3467007910514877),
 ('total_payments', 8.8667215371077752),
 ('shared_receipt_with_poi', 8.7464855321290802),
 ('loan_advances', 7.2427303965360181),
 ('expenses', 6.2342011405067401),
 ('from_poi_to_this_person', 5.3449415231473374),
 ('other', 4.204970858301416),
 ('from_poi_ratio', 3.2122622771285392),
 ('from_this_person_to_poi', 2.4265081272428781),
 ('director_fees', 2.1076559432760908),
 ('to_messages', 1.6988243485808501),
 ('deferral_payments', 0.2170589303395084),
 ('from_messages', 0.16416449823428736),
 ('restricted_stock_deferred', 0.06498431172371151)]

```

* 这些特征不需要进行特征缩放，因为进行测试的三个算法为：朴素贝叶斯、决策树、随机森林，特征缩放并不会显著影响这三个算法模型的结果，所以无需进行特征缩放。
* 根据邮件数据，我创建了两个新的特征"from\_poi\_ratio"：该员工收到来自嫌疑人的邮件数量占其收到邮件总数的比例，"to\_poi\_ratio"：该员工发送给嫌疑人的邮件数占其发送邮件总数的比例。这两个特征较好的说明了员工与嫌疑人之间的邮件往来情况，所以新建了这两个特征用于后续模型的测试。


3.你最终使用了什么算法？你还尝试了其他什么算法？不同算法之间的模型性能有何差异？【相关标准项：“选择算法”】

所有的算法测试时，均进行了特征选择和PCA步骤，最终使用了朴素贝叶斯算法，因其获得了最佳的结果。

我尝试的其他算法有：决策树和随机森林。

使用决策树时，其最佳参数一共使用了3个特征，其Recall的值非常不理想，远低于0.3，F1的值也小于0.3.

使用随机森林时，算法运行时间比较久，且每次的结果都不相同，这跟随机森林的原理有关，它的最佳参数使用了4个特征，它的结果稍好于决策树，但recall值也没有达到0.3的最低要求。

4.调整算法的参数是什么意思，如果你不这样做会发生什么？你是如何调整特定算法的参数的？（一些算法没有需要调整的参数 – 如果你选择的算法是这种情况，指明并简要解释对于你最终未选择的模型或需要参数调整的不同模型，例如决策树分类器，你会怎么做）。【相关标准项：“调整算法”】

调整算法的参数是指调整机器学习算法的参数，使之获得机器学习的最佳模型。如果不进行调参，可能部分算法无法获得最佳的模型，无法达到我们想要的结果。

本次测试的三个算法中，GaussianNB无需调整参数，而此次数据集的样本数量和特征都不算太多，所以DecisionTreeClassifier我们直接使用默认参数，下面介绍一下RandomForestClassifier的调参过程：
参数调整我使用了GridSearchCV方法，即通过网格搜索的方式进行调参，我们先定义好不同算法的参数范围，并使用网格搜索获得最佳参数，这里根据算法的特点以及本次数据集的情况，我们设定了随机森林如下参数范围：


* RandomForestClassifier:
```
        clf = RandomForestClassifier()
        clf_parameters = {algorithm + '__max_features':('auto','log2'),
            algorithm + '__n_estimators':[2, 5, 7, 10, 15],
            algorithm + '__min_samples_leaf': [1, 5, 10, 20],
            algorithm + '__min_samples_split': [2, 5, 10]}
```

最终通过网格搜索我们获得了随机森林的最佳参数分别为：

* RandomForestClassifier

```
{'RandomForest__max_features': 'auto', 'RandomForest__min_samples_leaf': 1, 'PCA__n_components': 2, 'SKB__k': 4, 'SKB__score_func': <function f_classif at 0x10fc7e578>, 'RandomForest__n_estimators': 2, 'RandomForest__min_samples_split': 5}

4 best features: ['bonus', 'exercised_stock_options', 'salary', 'total_stock_value']

```

5.什么是验证，未正确执行情况下的典型错误是什么？你是如何验证你的分析的？【相关标准项：“验证策略”】

验证是指我们将数据集分离成训练数据和测试数据，并使用训练数据来训练机器学习的算法模型，最终使用测试数据来验证我们获得的算法模型的性能。未正确执行验证的情况下可能会导致模型过拟合，以至于我们得到的模型没有较好的扩展性。

为了验证我的分析，我使用了test_classifier对我生成的模型进行验证，该方法中使用StratifiedShuffleSplit(labels, folds, random_state = 42)将数据集随机拆分成训练数据和测试数据，并代入指定的模型中，folds=1000表示重复过程1000次，最终计算Accuracy、Precision、	Recall、F1、F2的值，从而用来评估模型的性能。


6.给出至少 2 个评估度量并说明每个的平均性能。解释对用简单的语言表明算法性能的度量的解读。【相关标准项：“评估度量的使用”】

test_classifier返回的模型评估度量含义如下：

* Accuracy（准确率）：表示算法正确判断员工是否嫌疑人占总数的比例
* Precision（精准率）：表示算法成功判断员工为嫌疑人占所有判断为嫌疑人的比例
* Recall（召回率）：表示算法判断员工为嫌疑人占所有嫌疑人的比例
* F1：Precision 和 Recall的调和均值，可用来评估算法性能
* F2：F-measure一般化值，可用来评估算法性能
    
根据以上我们可以知道，当模型拥有较高的Precision值时，表明模型能较好的识别出嫌疑人。在测试的三个模型中，NavieBayes拥有最高的Precision值，高达0.55005，而DecisionTreeClassifier和RandomForestClassifier的Precision分别为0.36972和0.40837，所以在Precision方面，平均值为0.442:
NavieBayes 0.55005> RandomForestClassifier 0.40837>  DecisionTreeClassifier 0.36972

当模型拥有较高的Recall时，表明模型能找到更多的数据集中的嫌疑人的比例越高，在测试的三个模型中，平均值为0.242：
NavieBayes 0.30500 > RandomForestClassifier 0.21950>  DecisionTreeClassifier 0.20150

而F1作为Precision 和 Recall的调和均值，可以更全面的评估模型的性能，测试的三个模型的结果如下，平均值为0.313：
NavieBayes 0.39241 > RandomForestClassifier 0.28553>  DecisionTreeClassifier 0.26084

综合来看，NavieBayes模型的性能要显著高于另外两个模型，所以最终我选择了NavieBayes作为最佳模型。
  











 优达学城
2016年9月