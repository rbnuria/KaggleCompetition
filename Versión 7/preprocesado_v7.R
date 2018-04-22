###### EN ESTE ARCHIVO VAMOS A LLEVAR A CABO EL PREPROCESADO DE LOS DATOS

train = read.csv("../train.csv", header = TRUE, sep = ",")
test = read.csv("../test.csv", header = TRUE, sep = ",")

l_train <- train[length(train)]
d_train <- train[-length(train)]
data <- rbind(d_train, test)
id <- data[1]
data <- data[-1]

## Trabajamos sobre la versión 4: vamos a aplicar PCA pues la selección de características a mano no ha funcionado
## PCA obtiene el número de componentes (combinación lineal de las anteriormente obtenidas, que nos ofrecen un threshold introducido)
## Vamos a introducir threshold = 0.95 pues no nos importa que tenga muchos datos
library(DMwR)
library(caret)

## Aunque utilicemos predicción para el resto de valores perdidos, eliminamos aquellos que tienen demasiados
## No pueden estimarse el resto de valores
data['Alley'] <- NULL 
data['MiscFeature'] <- NULL 
data['Fence'] <- NULL 
data['PoolQC'] <- NULL 

#Estimamos el resto de valores perdidos
data <- knnImputation(data) #Valores por defecto


preprocess <- preProcess(data, method = "pca", thresh = 0.98)
predicted_data <- predict(preprocess, data)


predicted_data <- cbind(id, predicted_data)

train <- cbind(predicted_data[1:dim(train)[1],], l_train)
test <- predicted_data[(dim(train)[1]+1):dim(predicted_data)[1],]

#Guardamos en nuevos datasets
write.csv(train, file = "train.csv", sep = ",", row.names = FALSE)
write.csv(test, file = "test.csv", sep = ",", row.names = FALSE)