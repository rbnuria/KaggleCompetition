###### EN ESTE ARCHIVO VAMOS A LLEVAR A CABO EL PREPROCESADO DE LOS DATOS

train = read.csv("../train.csv", header = TRUE, sep = ",")
test = read.csv("../test.csv", header = TRUE, sep = ",")


## Versión 8: quitamos los valores perdidos pero intentamos hacerlo cambiando los parámetros del método
## knnImputation a ver si mejoramos los parámetros
#https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/knnImputation
library(DMwR)
library(CORElearn)
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

#Estimamos el resto de valores perdidos
noNA_train <- knnImputation(train, k = 30) ##Utiliamos 30 vecinos más cercanos
noNA_test <- knnImputation(test, k = 30) 

#Guardamos en nuevos datasets
write.csv(noNA_train, file = "notNA_train.csv", sep = ",", row.names = FALSE)
write.csv(noNA_test, file = "notNA_test.csv", sep = ",", row.names = FALSE)