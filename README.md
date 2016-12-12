# Mushroom-classifier
This repository quickly demonstrates the usage of varSelRF library in R to classify the "edible" and "poisonous" mushroom.

### Output from the model in the form of confusionMatrix 
Result from training data.
> confusionMatrix(training.pred, training.data$class, positive="p")

```R
Confusion Matrix and Statistics

          Reference
Prediction    e    p
         e 2946    0
         p    0 2742
                                     
               Accuracy : 1          
                 95% CI : (0.9994, 1)
    No Information Rate : 0.5179     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         
                                     
            Sensitivity : 1.0000     
            Specificity : 1.0000     
         Pos Pred Value : 1.0000     
         Neg Pred Value : 1.0000     
             Prevalence : 0.4821     
         Detection Rate : 0.4821     
   Detection Prevalence : 0.4821     
      Balanced Accuracy : 1.0000     
                                     
       'Positive' Class : p          
```                                
Result from testing data

> test.x <- subset(testing.data, select=var.sel$selected.vars)

> test.pred <- predict(var.sel$rf.model, test.x)

> confusionMatrix(test.pred, testing.data$class, positive="p")

```R
Confusion Matrix and Statistics
          Reference
Prediction    e    p
         e 1262    0
         p    0 1174
                                     
               Accuracy : 1          
                 95% CI : (0.9985, 1)
    No Information Rate : 0.5181     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         
                                     
            Sensitivity : 1.0000     
            Specificity : 1.0000     
         Pos Pred Value : 1.0000     
         Neg Pred Value : 1.0000     
             Prevalence : 0.4819     
         Detection Rate : 0.4819     
   Detection Prevalence : 0.4819     
      Balanced Accuracy : 1.0000     
                                     
       'Positive' Class : p   
```
### Variable Importances and other findings
Variable Importances: 
![alt text](https://github.com/netsatsawat/Mushroom-classifier/blob/master/VariableImp.jpeg)

No. of tree vs. error rate:
![alt text](https://github.com/netsatsawat/Mushroom-classifier/blob/master/Error%20ratio.jpeg)

## Acknowledgements
This dataset was originally donated to the UCI Machine Learning repository. You can learn more about past research using the data [here] (https://archive.ics.uci.edu/ml/datasets/Mushroom "UCI Mushroom").
