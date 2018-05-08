# Predictive Modelling Challenge
# Create a predictive model for the compressive strength of different concrete mixtures. 
# The training data set can be used to train the model. The test data set contains the mixtures 
# of which the compressive strength needs to be predicted. Write the compressive strength of the mixture 
# to a file with write.csv such that it only contains the index and the compressive strength. 
# This can be done by creating a data frame with the index and the compressive strength only 
# (say, e.g., predictions) and then issue the command

# Remark:
# The last part at bottom of this script (Boosting (GBM)) is the final model with the best performance
# GBM with parameters: shrinkage=0.10, depth=3, n/o trees=1000     
# RMSE=3.949497, R^2=0.9426465, MAE=2.763213	
# test set score: 12.6951

# The models tested were: Random Trees, SVM, NNET, PART and GBM

# -------------------------------------------
# Load the data set
library(readr)
setwd("C:/Somepath/")
concrete <- read_csv("trainingdata.csv")
#View(concrete)

library("beepr")
beep("ping")

#install.packages("doParallel")
# prerequisites:
# devtools::install_github('topepo/caret/pkg/caret')
library(doParallel)
registerDoParallel(cores = 2)

C = concrete
train = C[1:900,1:8]
trainresponse = C[1:900,9]

# -------- RANDOM FOREST -----------

# Start the clock
ptm <- proc.time()

library("caret")
control <- trainControl(method="cv", number=10)
set.seed(123)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(CompressiveStrength ~ ., data = C, method="rf", metric="RMSE", tuneGrid=tunegrid, trControl=control)
# Stop the clock
proc.time() - ptm
beep("ping")

# plot randomForest 
plot(rf_gridsearch)
rf_gridsearch
reprtree:::plot.getTree(rf_gridsearch$finalModel)
party::plot(rf_gridsearch, type="simple")

library(randomForest)
library(reprtree)
# model <- randomForest(CompressiveStrength ~ ., data=C, importance=TRUE, ntree=5, mtry = 5, do.trace=100)
model <- randomForest(CompressiveStrength ~ .,data=C,tunegrid=tunegrid,control=control)
reprtree:::plot.getTree(model)
library(party)
model <- ctree(CompressiveStrength ~ ., data=C)
plot(model, type="simple")

plot(CompressiveStrength ~ Age, data=C)

# visualisations
library(lattice)
levelplot(CompressiveStrength ~ Age * Water, C, cuts = 9, col.regions = rainbow(10)[10:1])
filled.contour(volcano, color = terrain.colors, asp = 1, plot.axes = contour(volcano, add = T))
persp(volcano, theta = 25, phi = 30, expand = 0.5, col = "lightblue")

# ------------------------------------
# Support Vector Machines
library(kernlab)
ctrl <- trainControl(method = "cv", number = 10)  # train control: cross validation, 10-folds
set.seed(123)

# Start the clock
ptm <- proc.time()
# svm train
svmFit <- train(CompressiveStrength ~ ., data = C, method = "svmRadial", tuneLength = 14, 
                preProc = c("center", "scale"), trControl = ctrl)
# Stop the clock
proc.time() - ptm
beep("ping")

svmFit
plot(svmFit)

# residuals vs. fitted
pred_train = predict(svmFit, train2)
plot(pred_train,col="Red")
points(train2$CompressiveStrength,col="Blue")

# ------------------------------------

ctrl <- trainControl(method = "cv", number = 10)  # train control: cross validation, 10-folds
nnetGrid <- expand.grid(.decay = c(0.4), .size = c(20), .bag = TRUE)
set.seed(123)

# Start the clock
ptm <- proc.time()
# nnet train
nnetFit <- train(CompressiveStrength ~ ., data = C, method = "avNNet", 
                 tuneGrid = nnetGrid, trControl = ctrl,
                 linout = TRUE, trace = FALSE, MaxNWts = 20 * (ncol(train) + 1) + 20 + 1,
                 maxit = 1000, preProc = c("center", "scale"))
# Stop the clock
proc.time() - ptm
beep("ping")

nnetFit
plot(nnetFit)

# residuals vs. fitted
pred_train = predict(nnetFit, C)
plot(pred_train,col="Red")
points(C$CompressiveStrength,col="Blue")

df_diag <- data.frame(residuals = nnetFit$finalModel$residuals, 
                      fitted = nnetFit$finalModel$fitted.values)
ggplot(data=C, aes(x=fitted, y=residuals)) + geom_point()

plot(nnetFit$finalModel$residuals,nnetFit$finalModel$fitted.values)

# Load the data set
library(readr)
testdata <- read_csv("C:/Somepath/testdata.csv")
#newdata = testdata[,2:9]
max(testdata$Index)

# construct test data
testset = testdata[,2:9]
testindex = testdata[,1]

# predict using randomn forest
pred_rf = predict(rf_gridsearch, testset)
#y1=cbind(testindex1,predictions1)

# predict using SVM
pred_svm = predict(svmFit, testset)
#y2=cbind(testindex2,predictions2)

# predict using NN
pred_nnet = predict(nnetFit, testset)

# write results
write.csv(as.data.frame(pred_rf),file=paste(path,"predictions_rf.csv",sep=""))
write.csv(as.data.frame(pred_svm),file=paste(path,"predictions_svm.csv",sep=""))
write.csv(as.data.frame(pred_nnet),file=paste(path,"predictions_nn.csv",sep=""))

# average results
pred_avg = (pred_rf+pred_nnet)/2
pred_avg
write.csv(as.data.frame(pred_avg),file=paste(path,"predictions_rfnn.csv",sep=""))

# reconstruct output
c(c(1,2,3),c(4,5,6))
c(c(y1$Index,y2$Index),c(y1$predictions1,y2$predictions2))

# visualise predictions
x=(C$CompressiveStrength)
y=(rep(predictions,7)[1:900])
plot(seq(min(x),max(x),length=900),x,col="Black")
points(seq(min(x),max(x),length=900),y,col="Blue")

# export predictions
path = "C:/Somepath/"
write.csv(as.data.frame(y1),file=paste(path,"predictions1.csv",sep=""))
write.csv(as.data.frame(y2),file=paste(path,"predictions2.csv",sep=""))

# visualisations   
summary(predictions)
summary(C$CompressiveStrength)
boxplot(C$CompressiveStrength,predictions,pred_train,pred_validate)

hist(C$CompressiveStrength)
hist(predictions)
hist(pred_train)
hist(pred_validate)

# attempt to visualise overfitting
length(C$CompressiveStrength)
length(rep(predictions,7))

x=(C$CompressiveStrength)
y=(rep(predictions,7)[1:900])
plot(seq(min(x),max(x),length=900),x,col="Black")
points(seq(min(x),max(x),length=900),y,col="Blue")

#-----------------------------
# Decision Tree
ctrl <- trainControl(
  method = "LGOCV", 
  repeats = 3, 
  savePred=TRUE,
  verboseIter = TRUE,
  preProcOptions = list(thresh = 0.95)
)   # train control (1)
ctrl <- trainControl(method = "cv", number = 10)  # train control (2): cross validation, 10-folds
preProcessInTrain<-c("center", "scale")
metric_used<-"RMSE"
model <- train(
  CompressiveStrength ~ ., data = C,
  method = "rpart",
  trControl = ctrl,
  metric=metric_used,
  tuneLength = 10,
  preProc = preProcessInTrain
)
library(rpart.plot)
rpart.plot(model$finalModel)

# ------------------------------------
# Boosting (GBM)

library("beepr")
indx <- createFolds(C$CompressiveStrength, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

library("gbm")
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(1000, 5000, by = 100),
                       n.minobsinnode = 1,
                       shrinkage = seq(0.08, 0.12, by = 0.01))
set.seed(100)

# Start clock
ptm <- proc.time()

gbmTune <- train(x = trainData, y = response,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl,
                 verbose = FALSE)
gbmTune
# Stop the clock
proc.time() - ptm
beep("ping")

# visualize
plot(gbmTune, auto.key = list(columns = 4, lines = TRUE))

gbmImp <- varImp(gbmTune, scale = FALSE)
plot(gbmImp)

# ------------------------------------
# predict testdata and export csv
pred_gbm = predict(gbmTune, testset)
write.csv(as.data.frame(pred_gbm),file="predictions_gbm.csv")

# ------------------------------------

