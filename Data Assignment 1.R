
### STAT 6302 Data Assignment 1 ###

# * = Indicates that certain things will not run until the df final.train is established


#Load packages
library(dplyr)
library(tidyr)
library(plyr)
library(ggplot2)
library(gmodels)
library(agricolae)
library(multcomp)
library(Sleuth2)
library(MASS)
library(car)
library(glmnet)
library(caret)
library(leaps)
library(bestglm)
library(VIM)
library("VIM")
library(forcats)
library(stringr)
library(WriteXLS)
library(splines)


#Read in dataset
train = read.csv("train.csv")
train = data.frame(train)
test = read.csv("test.csv")
test = data.frame(test)


#Drop variables
train.drop = train[,!(names(train) %in% c("Id","SalePrice"))]
test.drop = test[,!(names(test) %in% c("Id"))]

#Combine train and test set
house = rbind(train.drop, test.drop)


# Feature Engineering #

#Combine bathroom variables (Erik Bruin)
house$TotalBathrooms = house$FullBath + (house$HalfBath*0.5) + house$BsmtFullBath + (house$BsmtHalfBath*0.5)

#Create "Age" and 'Remodeling' variables (Erik Bruin)
house$Age = as.numeric(house$YrSold)-house$YearRemodAdd
house$Remodeling <- ifelse(house$YearBuilt==house$YearRemodAdd, 0, 1)
house$Remodeling[house$Remodeling == 0] <-"None"
house$Remodeling[house$Remodeling == 1] <-"Remodeling"

#Create New variable (Erik Bruin)
house$New <- ifelse(house$YrSold==house$YearBuilt, 1, 0)
table(house$New) #Way more old houses than new ones

#Keep only 'TotalBathrooms' from the bathroom variables
house = subset(house, select = -c(FullBath,HalfBath,BsmtFullBath,BsmtHalfBath))
str(house)


#Correlation between 'SalePrice' and bathroom variables

#SalePreice ~ FullBath
SalePrice = train$SalePrice
FullBath = train$FullBath
cor(FullBath, SalePrice) #r=0.5606638

#SalePreice ~ HalfBath
HalfBath = train$HalfBath
cor(FullBath, SalePrice) #r=0.5606638

#SalePreice ~ BsmtFullBath
BsmtFullBath = train$BsmtFullBath
cor(BsmtFullBath, SalePrice) #r=0.2271222

#SalePreice ~ BsmtHalfBath
BsmtHalfBath = train$BsmtHalfBath
cor(BsmtHalfBath, SalePrice) #r=-0.01684415


#See which variables have the greatest amount of NA values and see how to deal w/them
sapply(house, function(x) sum(is.na(x))) #Alley: 2721, PoolQC: 2909, Fence: 2348, MiscFeature: 2814, FireplaceQu: 1420

#Make 'Alley' into a factor to get rid of NAs
#Erik Bruin
house$Alley[house$Alley == "Grvl"] <-'Gravel'
house$Alley[house$Alley == "Pave"] <-'Paved'
house$Alley[is.na(house$Alley)] <- 'No Alley'
house$Alley <- as.factor(house$Alley)
str(house$Alley)

#For 'PoolQC,' get rid of NAs by changing 'NA' to 'None' and since ordinal change to integer
#Erik Bruin
house$PoolQC[is.na(house$PoolQC)] <- 'None'
Revalue <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
house$PoolQC<-as.integer(revalue(house$PoolQC, Revalue))
str(house$PoolQC)

#For 'Fence,' get rid of NAs by changing 'NA' to 'No Fence' and convert to factor
#Erik Bruin
house$Fence[is.na(house$Fence)] <- 'No Fence'
house$Fence <- as.factor(house$Fence)
str(house$Fence)

#For 'MiscFeature,' get rid of NAs by changing 'NA' to 'None' and convert to factor (there's no "Elev" value, so will become 'factor w/ 5 levels')
#Erik Bruin
house$MiscFeature[is.na(house$MiscFeature)] <- 'None'
house$MiscFeature <- as.factor(house$MiscFeature)
str(house$MiscFeature)

#For 'FireplaceQu,' get rid of NAs by changing 'NA' to 'None' and since ordinal change to integer
#Erik Bruin
house$FireplaceQu[is.na(house$FireplaceQu)] <- 'None'
house$FireplaceQu<-as.integer(revalue(house$FireplaceQu, Revalue))
str(house$FireplaceQu)

#These 5 variables don't have NA values anymore: Alley, PoolQC, Fence, MiscFeature, and FireplaceQu
sapply(house, function(x) sum(is.na(x)))


# Near Zero Variance #
near_zero = nearZeroVar(house)
near_zero
#The following variables have "near-zero variance" and are removed from the data
house = house[-c(5,6,8,9,11,14,22,31,35,36,39,45,48,51,59,60,63,64,65,66,67,68,70,71,79)]
str(house)
#Street,Alley,LandContour,Utilities,LandSlope,Condition2,RoofMatl,BsmtCond,BsmtFinType2,
#BsmtFinSF2,Heating,LowQualFinSF,KitchenAbvGr,Functional,GarageQual,GarageCond,OpenPorchSF,
#EnclosedPorch,X3SsnPorch,ScreenPorch,PoolArea,PoolQC,MiscFeature,MiscVal,New


#See which variables are numeric and impute by the mean
num = house[sapply(house,is.numeric)]
str(num)
num = kNN(num, k = 10, numFun = mean)
num = num[,-c(27:52)]

#Catregorical variables
categorical = house[!sapply(house,is.numeric)]
str(categorical)
categorical = kNN(categorical, k = 10, numFun = mode)
categorical = categorical[,-c(29:56)]

#Combine (no more NAs)
house = cbind(num, categorical)
sapply(house, function(x) sum(is.na(x)))
str(house) #54 variables


# Natural Cubic Spline #

#Plot to see for nonlinear relationship of SalePrice vs. GrLivArea
GrLivArea = train$GrLivArea
SalePrice = train$SalePrice
plot(GrLivArea, SalePrice, cex=0.5, col="darkgrey")

#Create a grid of values for 'GrLivArea' at which we want predictions
lims = range(GrLivArea)
GrLivArea.grid = seq(from=lims[1], to=lims[2])

#Fit
sp.fit = lm(train$SalePrice ~ ns(GrLivArea, df = 5), data = train)
sp.pred <- predict(sp.fit, newdata=list(GrLivArea=GrLivArea.grid), se=TRUE)
se.bands <- cbind(sp.pred$fit+2*sp.pred$se.fit, sp.pred$fit-2*sp.pred$se.fit)

#Plot spline
plot(GrLivArea, train$SalePrice, xlim=lims, cex=.5, col="darkgrey",
     main="Natural Cubic Spline",
     xlab="GrLivArea",
     ylab="SalePrice")
lines(GrLivArea.grid, sp.pred$fit, lwd=2, col="red")
matlines(GrLivArea.grid, se.bands, lwd=1, col="blue", lty=3)


#Create a natural spline with 5 df for the 'GrLivArea' column
spline = ns(house$GrLivArea, df=5)

#Combine the original data with the spline data
house = cbind(house, spline)

#Give the basis functions more reasonable column names
colnames(house)[55:59] = c('GrLivArea1', 'GrLivArea2', 'GrLivArea3',
                           'GrLivArea4', 'GrLivArea5')

#Drop original 'GrLivArea' column
drop = c('GrLivArea')
house = house[,!(names(house) %in% drop)]


#Make all variables numeric
house = data.frame(lapply(house, function(x) as.numeric(as.factor(x))))
str(house) #58 variables


#Train and test
set.seed(100)
train.index = nrow(train)
test.index = train.index + 1
total = train.index + nrow(test)

#log variables
house$LotFrontage = log(house$LotFrontage)
house$LotArea = log(house$LotArea)

#Create actual train and test sets again
final.train = house[1:train.index, ]
final.test = house[test.index:total, ]

#Reinstate SalePrice
final.train$SalePrice = train$SalePrice


#Correlation between SalePrice vs. TotalBathroom
SalePrice = final.train$SalePrice
TotalBathroom = final.train$TotalBathroom
cor(TotalBathroom, SalePrice) #0.6277554


#Plot showing negative correlation with 'Age' and 'SalePrice'
ggplot(data=final.train[!is.na(SalePrice),], aes(x=Age, y=SalePrice))+
  geom_point(col='black') + geom_smooth(method = "lm", se=FALSE, color="blue", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
  ggtitle("Scatterplot of SalePrice vs. Age") +
  theme(plot.title = element_text(hjust = 0.5))


#Make 'Remodeling' a character variable
final.train$Remodeling = as.character(final.train$Remodeling)
final.train$Remodeling[final.train$Remodeling == "1"] <-"None"
final.train$Remodeling[final.train$Remodeling == "2"] <-"Remodeling"

#Plot showing how remodeled homes are worth less
ggplot(final.train[!is.na(final.train$SalePrice),], aes(x=Remodeling, y=SalePrice, colour = factor(Remodeling), shape = factor(Remodeling))) +
  geom_bar(stat="identity", fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000)) +
  theme_grey(base_size = 18)

#Change back
final.train$Remodeling = as.numeric(final.train$Remodeling)


#Shows negative correlation of SalePrice vs. Age
cor(final.train$SalePrice[!is.na(final.train$SalePrice)], final.train$Age[!is.na(final.train$SalePrice)])


#Distribution of Sales Prices of Homes in Ames (not normally distributed)
ggplot(train, aes(x=SalePrice)) + geom_histogram(color="white", fill="black") +
  labs(title="Distribution of Sales Prices of Homes in Ames, Iowa",x="Sales Price", y = "Frequency")+
  theme_classic() #not normally distributed

#Take the log of SalePrice
final.train$SalePrice = log(train$SalePrice)


#Backwards variable selection (specify 10-fold cross-validation first)
train.control <- trainControl(method = "cv", number = 10)

back.model <- train(SalePrice ~ ., data = final.train,
                    method = "leapBackward",
                    tuneGrid = data.frame(nvmax = 1:18),
                    trControl = train.control)

back.model$bestTune
back.model$results[18,] #RMSE
summary(back.model$finalModel)
coef(back.model$finalModel, 18)


#Forward variable selection
forward.model <- train(SalePrice ~ ., data = final.train,
                       method = "leapForward", 
                       tuneGrid = data.frame(nvmax = 1:18),
                       trControl = train.control)

forward.model$bestTune
forward.model$results[18,] #RMSE
summary(forward.model$finalModel)
coef(forward.model$finalModel, 18)


#Stepwise variable selection
step.model <- train(SalePrice ~ ., data = final.train,
                    method = "leapSeq", 
                    tuneGrid = data.frame(nvmax = 1:18),
                    trControl = train.control)

step.model$bestTune
step.model$results[18,] #RMSE
summary(step.model$finalModel)
coef(step.model$finalModel, 18)


#Compare variable selection methods' RMSEs

#Backward
back.model$results[18,]
#Forward
forward.model$results[18,]
#Stepwise
step.model$results[18,]

#Stepwise has the best RMSE, so we will use the variables selected from it for penalized regression
coef(step.model$finalModel, 18)

final.train = final.train %>% dplyr::select(LotArea,OverallQual,OverallCond,YearBuilt,BsmtFinSF1,TotalBsmtSF,X2ndFlrSF,FireplaceQu,GarageCars,TotalBathrooms,BsmtExposure,CentralAir,Electrical,GarageType,Fence,Remodeling,GrLivArea3,GrLivArea4,SalePrice)
final.test = final.test %>% dplyr::select(LotArea,OverallQual,OverallCond,YearBuilt,BsmtFinSF1,TotalBsmtSF,X2ndFlrSF,FireplaceQu,GarageCars,TotalBathrooms,BsmtExposure,CentralAir,Electrical,GarageType,Fence,Remodeling,GrLivArea3,GrLivArea4)


#Based off of variables chosen from stepwise variable selection
model = lm(SalePrice ~ LotArea+OverallQual+OverallCond+YearBuilt+BsmtFinSF1+TotalBsmtSF+X2ndFlrSF+FireplaceQu+GarageCars+TotalBathrooms+BsmtExposure+CentralAir+Electrical+GarageType+Fence+Remodeling+GrLivArea3+GrLivArea4, data=final.train)
summary(model)

#Check VIF scores to see which variables have inflated standard errors
vif(model)
res = cor(final.train) #correlation between variables


# Penalized Regression Methods #

#Ridge Regression (note that cv.glmnet by default does 10-fold cross-validation)
rr.glmnet = cv.glmnet(as.matrix(final.train[1:18]), final.train$SalePrice, alpha=0)
attributes(rr.glmnet)
best.lambda <- rr.glmnet$lambda.min #Optimal tuning parameter
ridge.coef = coef(rr.glmnet, s=best.lambda)
#RMSE
sqrt(rr.glmnet$cvm[rr.glmnet$lambda == rr.glmnet$lambda.1se])
#Predict
ridge.pred = predict(rr.glmnet, as.matrix(final.test), s=best.lambda)


#Lasso
lasso.glmnet = cv.glmnet(as.matrix(final.train[,1:18]), final.train$SalePrice, alpha=1)
attributes(lasso.glmnet)
best.lambda2 = lasso.glmnet$lambda.min #Optimal tuning parameter
lasso.coef = coef(lasso.glmnet, s=best.lambda2)
lasso.coef[lasso.coef !=0]
#RMSE
sqrt(lasso.glmnet$cvm[lasso.glmnet$lambda == lasso.glmnet$lambda.1se])
#Predict
lasso.pred = predict(lasso.glmnet, newx=as.matrix(final.test), s=best.lambda2)


#Elastic Net
tcontrol = trainControl(method="repeatedcv", number=10, repeats=5)

en.glmnet = train(as.matrix(final.train[,1:18]), final.train$SalePrice, trControl=tcontrol,
                  method="glmnet", tuneLength=10)
attributes(en.glmnet)
en.glmnet$results
en.glmnet$bestTune #Optimal tuning parameter
en.glmnet2 = en.glmnet$finalModel
en.coef = coef(en.glmnet2, s=en.glmnet$bestTune$lambda)
#RMSE
min(en.glmnet$results$RMSE)
#Predict
en.pred = predict(en.glmnet2, as.matrix(final.test), s=en.glmnet$bestTune$lambda)


#Solution paths for penalized regression
par(mfrow=c(2,2))

#Ridge
ridge.mod = glmnet(as.matrix(final.train[,1:18]), final.train$SalePrice, alpha=0)
plot(ridge.mod)
title("Ridge Regression Solution Path", line = 2.5)
#Lasso
lasso.mod = glmnet(as.matrix(final.train[,1:18]), final.train$SalePrice, alpha=1)
plot(lasso.mod)
title("Lasso Solution Path", line = 2.5)
#EN
plot(en.glmnet2)
title("Elastic Net Solution Path", line = 2.5)


#Compare penalized regression models' RMSEs and Kaggle Scores

#Ridge
sqrt(rr.glmnet$cvm[rr.glmnet$lambda == rr.glmnet$lambda.1se]) #Kaggle Score = 0.14228
#Lasso
sqrt(lasso.glmnet$cvm[lasso.glmnet$lambda == lasso.glmnet$lambda.1se]) #Kaggle Score = 0.13987
#EN
min(en.glmnet$results$RMSE) #Kaggle Score = 0.14013

#Elastic Net has the lowest RMSE but Lasso has the best Kaggle score, so I'll use Lasso


#Final model
lasso.coef #'Electrical' and 'GarageType' are removed


#Write Ridge Predictions as Excel file
WriteXLS(data.frame(ridge.pred))

#Write Lasso Predictions as Excel file
WriteXLS(data.frame(lasso.pred))

#Write EN Predictions as Excel file
WriteXLS(data.frame(en.pred))

#I converted the log values in Excel before submitting to Kaggle


