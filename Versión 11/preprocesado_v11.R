###### EN ESTE ARCHIVO VAMOS A LLEVAR A CABO EL PREPROCESADO DE LOS DATOS

train = read.csv("../train.csv", header = TRUE, sep = ",")
test = read.csv("../test.csv", header = TRUE, sep = ",")


## Versión 11: predecimos los valores perdidos con mice -> utilizamos mice.rf para utilizar random forest
#https://www.r-bloggers.com/missing-value-treatment/
#https://www.rdocumentation.org/packages/mice/versions/2.46.0/topics/mice.impute.rf
library(mice)

## Aunque utilicemos predicción para el resto de valores perdidos, eliminamos aquellos que tienen demasiados
## No pueden estimarse el resto de valores
train['Alley'] <- NULL 
train['MiscFeature'] <- NULL 
train['Fence'] <- NULL 
train['PoolQC'] <- NULL 

test['Alley'] <- NULL 
test['MiscFeature'] <- NULL 
test['Fence'] <- NULL 
test['PoolQC'] <- NULL 


rf_train <- mice(train, meth = "rf", ntree = 15)
rf_test <- mice(test, meth = "rf", ntree = 15)

noNA_train <- complete(rf_train)
noNA_test <- complete(rf_test)
#Guardamos en nuevos datasets
write.csv(noNA_train, file = "notNA_train.csv", sep = ",", row.names = FALSE)
write.csv(noNA_test, file = "notNA_test.csv", sep = ",", row.names = FALSE)