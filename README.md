# Kaggle/Santander Product Recommendation
<div align="center">
  <img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/5558/media/santander-banner-ts-660x.png"><br><br>
</div>
## Abstract
- Host : **Santander**, British bank, wholly owned by the Spanish Santander Group.
- Prize : $ 60,000
- Problem : Multi-class Classification based Recommendation
- Evaluation : [MAP@7](https://www.kaggle.com/wiki/MeanAveragePrecision)
- Period : Oct 26 2016 ~ Dec 21 2016 (66 days)

**Santander Bank** offers a lending hand to their customers through personalized product recommendations. In their second competition, Santander is challenging Kagglers to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers.

Competition data consists of customer data from 2015-01 ~ 2016-05 (total of 17 month timestamps) including customer's demographic information and their product purchase behavior. Competition challenges you to predict top 7 products out of 24, that each customer in the test data is most likely to purchase on 2016-06.

Evaluation metric is in MAP@7, which made the direct optimization difficult during training phase. Instead, the mlogloss was widely used among kagglers to indirectly optimize the solution.

With BreakfastPirates generous sharing, using 2015-06 data-only as a training data seemed to perform pretty well in the leaderboard (reaching almost ~0.03). Single model performance was enough to place you on top of the leaderboard, since MAP@7 made the effect of ensemble relatively weak.

As always, feature engineering seemed to be the most important factor in this competition, along with good cv scheme to reach the best hyper-parameter that squeezes the performance from the given data.


## Result
| Submission | CV LogLoss | Public LB | Rank | Private LB | Rank |
|:----------:|:----------:|:---------:|:----:|:----------:|:----:|
| bare_minimum | 1.84295 | 
| kweonwooj | 
| 84th | 
| 14th | 
| 7th | 
| 6th | 
| 1st| 

## How to Run

**[Data]** 

Place data in ```root_input``` directory. You can download data from [here](https://www.kaggle.com/c/santander-product-recommendation/data).

**[Code]**

Above results can be replicated by runinng

```
python code/main.py
```
for each of the directories.

Make sure you are on Python 3.5.2 with library versions same as specified in requirements.txt

**[Submit]**

Submit the resulting csv file [here](https://www.kaggle.com/c/santander-product-recommendation/submissions/attach) and verify the score.

## Expected Result

img of submitted result

## Winnig Solutions
- 6th place solution on [Forum](https://www.kaggle.com/c/santander-product-recommendation/forums/t/26786/solution-sharing) by BreakfastPirate
- 7th place solution on [Forum](https://www.kaggle.com/c/santander-product-recommendation/forums/t/26802/7-solution) by Evgeny Patekha
- 14th place solution on [Forum](https://www.kaggle.com/c/santander-product-recommendation/forums/t/26785/aj-and-matt-s-solution-details), [Blog](http://alanpryorjr.com/Kaggle-Competition-Santander-Solution/), [GitHub](https://github.com/apryor6/Kaggle-Competition-Santander) by Alan (AJ) Pryor, Jr.
- 84th place solution on [forum](https://www.kaggle.com/c/santander-product-recommendation/forums/t/26789/simple-model-solution-0-0305221-top-5) by MxDbld