# Fraud Detection

The task is to predict fraud transaction based on a transaction history. The main focus is on minimizing false negative forecasts and not increasing false positive forecasts too much.

## Data
[The original dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) is a transaction with numerical, categorical and time features. The positive class distribution is $\approx0.58$%.
 
**Created Features**:
* The dates were divided into hours, day_of_week, month with created cyclic variants (sin/cos) and flags 'is night' and 'is weekend';
* Frequency of transactions for previous periods (1 minute, 5 minutes, 1 hour, 1 day, 7 days);
* Flags 'is new state', 'is new city', 'is new state' indicate that the transaction is being made in a new location;
* Type of merchant category  ('pos' and 'net'), flag 'is favorite category' indicate that the category is different from the mode for a credit card;
* Distances to the median latitude and longitude;
* Age at the time of the transaction;
* Median credit card amount.
  
**Deleted Fearutes:**
* Credit card number and transaction number;
* Time of the transaction;
* Name;
* ZIP.

## Model
CatBoostClassifier trained on GPU.

Hyperparameter optimization with optuna (depth, learning rate, l2 leaf regularization, bagging temperature, random strength, border cound).

An early stop was used.

## Threshold and Metrics

The main metric is AUC-ROC for model selection.

The threshold was selected based on the cost-based metric.
$$\ln(\text{FN amounts + 1}) \times 5 + \ln(\text{FP amounts + 1}) \times 0.5$$

The logarithm of the sum is used to reduce the impact of extreme transactions on the cost metric.

The final threshold is $\approx0.06$, which minimizes FN while keeping FP low.

## Results

$TP \approx1890$, $FN \approx255$ catch $\approx88$% of fraudulent transactions.

$FP \approx668$, $TN >550000$ → there are almost no false predictions.

Final metrics:
* ROC AUC: 0.9978,
* F1: 0.8000,
* Balanced Accuracy: 0.9399,
* Cost: 7076.7388.

The graphs are built for:
* ROC-AUC,
* Precision-Recall curve,
* FP/FN by threshold,
* Cost metric by threshold.

## Conclusion

Pipeline has shown high efficiency on an unbalanced dataset.

Cost-aware threshold tuning and logical sums allow you to control business risks and the ratio of false predictions.

Possible improvements: processing of rare outliers, additional user-specific aggregates, and model ensembles.