###### EN ESTE ARCHIVO VAMOS A LLEVAR A CABO EL PREPROCESADO DE LOS DATOS

train = read.csv("../train.csv", header = TRUE, sep = ",")
test = read.csv("../test.csv", header = TRUE, sep = ",")


## En esta primera versión vamos a tratar los valores perdidos
#https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/knnImputation
library(DMwR)

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
noNA_train <- knnImputation(train) #Valores por defecto
noNA_test <- knnImputation(test) #Valores por defecto

#Guardamos en nuevos datasets
write.csv(noNA_train, file = "notNA_train.csv", sep = ",", row.names = FALSE)
write.csv(noNA_test, file = "notNA_test.csv", sep = ",", row.names = FALSE)