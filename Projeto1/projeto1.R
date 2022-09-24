# Projeto 1 - Machine Learning em Logística Prevendo o Consumo de Energia de Carros Elétricos
# Gabriel Araujo Carlos

setwd("C:/Users/Yoh/Desktop/R/Projeto1")
getwd()

# Pacotes
library(dplyr)
library(ggplot2)
require(randomForest)
library(tidyr)
library(MLmetrics)
library(e1071)
library(caret)

# Carrega o dataset
dados <- read.csv('dataset.csv',header=T,sep=";",na.strings=c(""," ","NA"))

nomesColunas=c("CarFullName","Make","Model","MinimalPrice","EnginePower","MaximumTorque","TypeBrakes","DriveType","BatteryCapacity","Range","Wheelbase","Length","Width","Height","MinimalEmptyWeight","PermissibleGrossWeight","MaximumLoadCapacity","NumberSeats","NumberDoors","TireSize","MaximumSpeed","BootCapacity","Acceleration","MaximumDCChargingPower","MeanEnergyConsumption")
colnames(dados) <- nomesColunas
# Visualiza os dados
View(dados) 

# Verificando ocorrência de valores NA
colSums(is.na(dados))

# Remover valores missing para feature selection.
dadosSelection <- na.omit(dados)

#Transformar as variaveis em numericas
dadosSelection <- dadosSelection %>% 
  mutate(BatteryCapacity = as.numeric(gsub(",", ".",BatteryCapacity)))%>% 
  mutate(Wheelbase = as.numeric(gsub(",", ".",Wheelbase)))%>% 
  mutate(Length = as.numeric(gsub(",", ".",Length)))%>% 
  mutate(Width = as.numeric(gsub(",", ".",Width)))%>% 
  mutate(Height = as.numeric(gsub(",", ".",Height)))%>% 
  mutate(Acceleration = as.numeric(gsub(",", ".",Acceleration)))%>% 
  mutate(MeanEnergyConsumption = as.numeric(gsub(",", ".",MeanEnergyConsumption)))
#Transformar as variaveis em categóricas
dadosSelection <- dadosSelection %>% 
  mutate(Make = as.factor(Make)) %>% 
  mutate(TypeBrakes = as.factor(TypeBrakes)) %>% 
  mutate(DriveType = as.factor(DriveType))

# Tipos de dados
str(dadosSelection)

#Feature Selection
modelo <- randomForest(MeanEnergyConsumption ~ . , 
                       data = dadosSelection, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE)
varImpPlot(modelo)


##### Decidi utilizar as 10 melhores variaveis baseado no incNodePurity##### 


dadosFilter = dados[c("Make","Wheelbase","PermissibleGrossWeight","MinimalPrice","MinimalEmptyWeight","MaximumTorque","EnginePower","Width","DriveType","Length","MeanEnergyConsumption")]
View(dadosFilter)
str(dadosFilter)
#Transformar as variaveis
dadosFilter <- dadosFilter %>% 
  mutate(Make = as.factor(Make)) %>% 
  mutate(DriveType = as.factor(DriveType)) %>% 
  mutate(Wheelbase = as.numeric(gsub(",", ".",Wheelbase)))%>% 
  mutate(Length = as.numeric(gsub(",", ".",Length)))%>% 
  mutate(Width = as.numeric(gsub(",", ".",Width)))%>% 
  mutate(MeanEnergyConsumption = as.numeric(gsub(",", ".",MeanEnergyConsumption)))


##### Como irei utilizar um metodo de aprendizagem supervisionada, observações sem informação de 
##### variável alvo nao são uteis.Assim irei remove-las.


dadosFilter <- dadosFilter %>% drop_na(MeanEnergyConsumption)

# Dividir os dados entre treino e teste
dadosFilter$id <- 1:nrow(dadosFilter)
train <- dadosFilter %>% dplyr::sample_frac(0.7)
test  <- dplyr::anti_join(dadosFilter, train, by = 'id')

# Criando modelo de regressao Linear
modeloLinerRegression = lm(MeanEnergyConsumption ~ .-id,data = train)
summary(modeloLinerRegression)
# predict(modeloLinerRegression,newdata = test) - o modelo apresenta erro para previsao devido ao fato 
# da variavel make ter valores diferentes entre o treino e teste


##### Por ter muitos valores diferentes e não ter dados suficientes,
##### a variavel Make atrapalha a analise.Assim irei remove-la do modelo.


dadosFilter = dadosFilter[c("Wheelbase","PermissibleGrossWeight","MinimalPrice","MinimalEmptyWeight","MaximumTorque","EnginePower","Width","DriveType","Length","MeanEnergyConsumption")]
#refazendo o processo de divisao entre treino e teste, e criação do modelo
dadosFilter$id <- 1:nrow(dadosFilter)
train <- dadosFilter %>% dplyr::sample_frac(0.7)
test  <- dplyr::anti_join(dadosFilter, train, by = 'id')
train$id <- NULL
test$id <- NULL
dadosFilter$id <- NULL
modeloLinerRegression = lm(MeanEnergyConsumption ~ .,data = train)
summary(modeloLinerRegression)


##### o modelo diminui o valor de R-squared, podendo ser uma melhora na generalização(diminuindo o overfitting).


predict(modeloLinerRegression,newdata = test[,-(10)])
R2_Score(y_pred = predict(modeloLinerRegression,newdata = test[,-(10)]), y_true = test$MeanEnergyConsumption)
#0.570068
# Testando segundo algoritmo
modelsvm = svm(MeanEnergyConsumption ~ .,data = train)
R2_Score(y_pred = predict(modelsvm,newdata = test[,-(10)]), y_true = test$MeanEnergyConsumption)
#0.951608
# Testando um terceiro algoritmo
modelKnn = knnreg(MeanEnergyConsumption ~ .,data = train)
R2_Score(y_pred = predict(modelKnn,newdata = test[,-(10)]), y_true = test$MeanEnergyConsumption)
#0.889887


##### O  modelo que utiliza a versao de regressao do SVM apresentou uma melhora consideravel 
##### na performance(utilizando R-squared).
##### Assim opto por seguir os trabalhos com ele.Tentarei melhorar a performance com alterações.
##### Como o dataset nao apresenta outliers, irei utilizar o min-max para normalizar os dados


# Criar funcao para aplicar a tecnica de min-max
minMax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#Verificando quais as variaveis numericas
numeric_variable_list <- sapply(dadosFilter, is.numeric)
# Normalizando os dados
dadosNormalizados <- as.data.frame(lapply(dadosFilter[numeric_variable_list], minMax))
# Acrescentando a variavel nao numerica
dadosNormalizados$DriveType = dadosFilter$DriveType

#Dividindo o dataframe entre treino e teste
train <- dadosNormalizados %>% dplyr::sample_frac(0.7)
test  <- dplyr::anti_join(dadosNormalizados, train, by = 'id')
train$id <- NULL
test$id <- NULL

#Criando e validando o novo modelo
modelsvm2 = svm(MeanEnergyConsumption ~ .,data = train)
R2_Score(y_pred = predict(modelsvm2,newdata = test[,-(9)]), y_true = test$MeanEnergyConsumption)


##### O novo modelo com dados normalizados apresentaram um desempenho pior.
##### Assim o modelo final sera com os dados nao normalizados, utilizando 9 variaveis.


modelsvm
##### Verificando que os valores que indiquei nao sao reproduziveis tentei criar utilizar o H2o para 
##### poder deixar o modelo mais performatico disponivel.Porem os algoritmos disponiveis nao aceitam
##### um dataset com poucas observacoes.