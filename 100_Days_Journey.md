# 100 Days Note

* Day1-13: Data Preprocessing
  
  Day14- : EDA
  
* Day 1
  
  * Discovery journey
    * Find the problem -> prototype solution -> improve -> sharing -> practice -> participate competition
  
* Day2
  * Three types of ML
    * Supervised Learning, Unsupervised Learning, Reinforcement Learning
  
* Day3
  * ML Process	
    * Preprocessing: missing value, outlier, standardisation
    * Define goal: regression/classification problem, predictor and target value
    * Evaluation: regression -> RMSE, MAE, R-square; classification -> Accuracy, F1 score, AUC
    * Build and Tune Model

* Day4

  * Explore Data Analysis (EDA)

    <img src='Screenshots/data_analysis_process.png'>

* Day5

  * Read and write file

  * `ndarray`, a fast and space-efficient multidimensional array providing vectorized arithmetic operations

    <img src='Screenshots/data_loading_efficiency.png'>

* Day6
  * Data Preprocessing
  * Label Encoding vs One hot encode
  
* Day7

  * Variable types: numerical & categorical

* Day8

  * EDA
    * mean, median, mode
    * min, max, range, quartiles, variance, standard deviation

* Day9
  * Outlier
    * How to check?
      * check summary statistics (avg, sd, median, IQR)
      * plot: boxplot for univariate feature and scatterplot for multivariate
    * How to deal with outlier: replace with median/average, delete all columns, create new columns, etc

* Day10
  * Outlier with code two methods
    * change the range (set min and max and force the outlier to be min value or max value)
    * directly exclude the data
* Day11
  * whether we should normalise the continuous data?
    * For regression model: affect
    * For tree-based model: no effect
* Day12
  * standard scaler vs minmax scaler
    * standard scaler -> not easy influence by outlier
    *  minmax scaler -> easy influence by outlier -> suitable for data already deal with outlier
* Day13
  * Dataframe manipulation
    * column to row: pd.melt(df)
    * row to column: pd.pivot()
    * regex filter: `df.filter(regex=)`
* Day14 EDA
  * Correlation Coefficient: -1~1 and measure two random variables linear relationship
    * use scatter plot to visualise the relationship
  * np.corrcoef
* Day15 
  * Correlation
  * df.corr()
* Day16 
  * Kernel density estimation -> computation intensive
  * plt.style.use(‘default’) # 不需設定就會使⽤用預設
    plt.style.use('ggplot')
    plt.style.use(‘seaborn’) # 或採⽤用 seaborn 套件繪圖
  * [Python graph gallery](<https://python-graph-gallery.com/>)
  * [KDE vs Histogram](<https://blog.csdn.net/unixtch/article/details/78556499>)
    * Histogram will bias if the bin setting is not accurate -> KDE no this issue
    * Histogram distribution is not smooth

* Day17 Discretising (連續型變數離散化)

  * reduce the impact of outlier
  * 等寬劃分: pd.cut
  * 等頻劃分: pd.qcut

* Day18 Discretising

  * discretising + groupby to see the trend

* Day19 Subplot

  * sns.jointplot

* Day20 Pairplot, Heatmap

  * sns.pairplot(df, hue = 'continent')

* Day21 Logistic Regression

  * remove column appear in test not train -> app_train, `app_test = app_train.align(app_test, join = 'inner', axis = 1)`

* Day22 Feature Engineer

  * count encoding -> sensitive to outlier

  * label count encoding -> not sensitive to outlier and also rank category by count (combination of label encoding and count encoding)

  * target encoding ->  Encode categorical variables by their ratio of target -> avoid overfit

  * box-cox transformation -> solve normality problems (non-normal data)

    

  <img src= "screenshots/feature_engineer.png">

  

  <img src="screenshots/ml_process.png">

* Day23 Remove Skewness
  * purpose of remove skewness -> make the distribution toward normality
  * Methods: take log, sqrt, boxcox
    * Boxcox: 使⽤用 box-cox 分布去偏時，除了了注意 λ 參參數要介於 **0**
      到 **0.5** 之間 (0=log transform, 0.5 = sqrt)，並且要注意轉換前的數值不可⼩小於等於 **0** (if <=0, then add 0.01-1 to original value)

* Day24 Categorical variables processing - label encoding and one-hot encode

  * auto generate mapping code: 

    class_mapping = {label:idx for idx,label in enumerate(set(df['class label']))}

  * one hot encoding / label encoding

  * Tree model no need use one hot encode using label encoding is enough since they don't treat number as ordering (ex:2>1)

  * Logistic regression: one hot encoding better; GradientBoostingRegressor/linear regression: label encoding -> but different dataset might have diff results

  * Pros and Cons

    <img src="screenshots/categorical_variable_encoding.png">

* Day25 Categorical variables processing - Mean encoding
  * 我们可以尝试使用**平均数编码（mean encoding）**的编码方法，在贝叶斯的架构下，利用所要预测的应变量（target variable），有监督地确定最适合这个定性特征的编码方式。在Kaggle的数据竞赛中，这也是一种常见的提高分数的手段。
  * when to use: when the variable highly correlated with the target or when a variable that has many categories
  * powerful encoding methods but probably will overfit (use smooth methods) -> use cross validation check score

* Day 26  Categorical variables processing - Count encoding, feature hash

  * 計數編碼是計算類別在資料中的出現次數，當⽬目標平均值與類別筆數呈正/負相關時，可以考慮使⽤用
  * Can apply together with label encoding
  * 觀察欄位相異值數量:`df.select_dtypes(include=["object"]).apply(pd.Series.nunique)`

* Day27 Time series features

  <img src="screenshots/time_series_processing.png">

  <img src="screenshots/time_series_processing2.png">

  * strptime (將⽂文字解析成時間物件), strftime (轉出時間物件各欄欄位成⽂文字) for datetime package
    * datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") -> change to string
    * datetime.datetime.strptime("2014-12-31 18:20:10", "%Y-%m-%d %H:%M:%S") -> change to datetime format