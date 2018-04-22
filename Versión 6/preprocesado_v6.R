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
train <- knnImputation(train) #Valores por defecto
test <- knnImputation(test) #Valores por defecto

## Elegimos las variables numéricas para la matriz de correlación-> 
numeric_attr <- train[c(2,4,5,17,18,19,20,26,34,36,37, 43,44, 46, 47, 48, 49, 50, 51, 61, 66, 67, 68, 69, 70, 71, 72, 73, 74,77)]

corr_train <- cor(numeric_attr, use = "complete.obs") 
#print(corr_train)

### Eliminamos aquelas variables que tengan corr con salePrice < 0.3
train <- train[-c(2, 18, 36, 37, 48, 50, 51, 68, 69, 70, 71, 72, 73, 74)]
test <- test[-c(2, 18, 36, 37, 48, 50, 51, 68, 69, 70, 71, 72, 73, 74)]

### Buscamos ahora variables altamente correladas, para poder eliminar una de ellas (> 0.9)-> no vemos a simple vista, devolvemos estos datos

#Guardamos en nuevos datasets
write.csv(train, file = "train.csv", sep = ",", row.names = FALSE)
write.csv(test, file = "test.csv", sep = ",", row.names = FALSE)
