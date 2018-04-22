###### EN ESTE ARCHIVO VAMOS A LLEVAR A CABO EL PREPROCESADO DE LOS DATOS

train = read.csv("../train.csv", header = TRUE, sep = ",")
test = read.csv("../test.csv", header = TRUE, sep = ",")

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


## Versión 5: además de la eliminación de datos perdidos anterior, vamos a centrar y escalar, método preProcess vale pa to
#https://www.rdocumentation.org/packages/caret/versions/6.0-78/topics/preProcess
library(caret)
library(DMwR)

#Estimamos el resto de valores perdidos
noNA_train <- knnImputation(train) #Valores por defecto
noNA_test <- knnImputation(test) #Valores por defecto

preprocess_1 <- preProcess(noNA_train[,2:(length(noNA_train)-1)], method = c("scale", "center")) #range por defecto en 0-1
preprocess_2 <- preProcess(noNA_test[,2:(length(noNA_test))], method = c("scale", "center"))

pred_train <- predict(preprocess_1, noNA_train[,2:(length(noNA_train)-1)])
pred_test <- predict(preprocess_2, noNA_test[,2:(length(noNA_test))])

new_train <- noNA_train
new_train[,2:(length(new_train)-1)] <- pred_train

new_test <- noNA_test
new_test[,2:(length(new_test)-1)] <- pred_test


#Guardamos en nuevos datasets
write.csv(new_train, file = "train.csv", sep = ",", row.names = FALSE)
write.csv(new_test, file = "test.csv", sep = ",", row.names = FALSE)
