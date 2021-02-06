

#1. Load MNIST data
#Downloaded and unzipped the MNIST dataset
#Downloaded LoadMNIST.r and changed the PATH to the directory that holds the unzipped MNIST files
#SETWD to the location where LoadMNIST.r is saved,in my case 
setwd('C:/Users/johnr/OneDrive/IML Coursework 1') 
source('LoadMNIST.r') # Runs the script in R to create the objects containing the MNIST dataset detailed in the loading file

#2. Install and load relevant packages
# install.packages("class", "e1071", "caret", "factoMinerR", "factoextra", "nnet", "RSNNS", "ggfortify","Rtsne")
library(class)
library(caret)
library(FactoMineR)
library(factoextra)
library(nnet)
library(e1071)
library(RSNNS)
library(ggfortify)
library(Rtsne)

#3. prep data, scale train and test sets
train_labels <- train$y
test_labels <- test$y
train_x <- train$x
train_scaled <-
  as.data.frame(scale(train_x, scale = FALSE, center = TRUE))
test_x <- test$x
test_scaled <-
  as.data.frame(scale(test_x, scale = FALSE, center = TRUE))

#Caret and GGplot will require the labels as a factor, so we format them again
train_labels_factor <- as.factor(train_labels)

#Examine data and visualise the dataset with GGplot
tsne_out <- Rtsne(train$x)
tsne_plot <-
  data.frame(x = tsne_out$Y[, 1], y = tsne_out$Y[, 2], col = train_labels_factor)
ggplot(tsne_plot) + geom_point(aes(x = x, y = y, color = col))



#4. Perform PCA and review PC's
trainingPCA <- prcomp(train_scaled)
summary(trainingPCA)

#Visualize the PCA. Lets plot the PC's in 2d and see if we can make sense of it
#clearly there are too many components (754) to visualise it clearly this way
fviz_pca_ind(
  trainingPCA,
  geom.ind = "point",
  col.ind = train_labels_factor,
  palette = c(
    "#00AFBB",
    "#E7B800",
    "#FC4E07",
    "#A4A4A4",
    "#CC6666",
    "#9999CC",
    "#66CC99",
    "#0066CC",
    "#990000",
    "#B5B4B4"
  ),
  addEllipses = TRUE,
  legend.title = "Groups"
)


#Can we select the best number of components with scree plot?
fviz_eig(
  trainingPCA,
  addlabels = TRUE,
  ylim = c(0, 50),
  ncp = 25
)
#We can see that it does not display easily. We are missing a lot of dimensions
#there are two many components to display, so let's review the eigenvalues instead
eigen_val_trainingPCA <- get_eig(trainingPCA)
eigen_val_trainingPCA
#Reviewing this we can see that 48 components gives us 81% of the variances contained in the data.


rotate <-  trainingPCA$rotation[, 1:48]
final_Train <- as.matrix(train_scaled) %*% (rotate)
final_Test <- as.matrix(test_scaled) %*% (rotate)

#5. PCA Reconstruction
#is 48 PC's sufficent? How well can we reconstruct the original data from this?
#Visualise your PCA reconstruction and explain why this is sufficient

numComponents <- 48
reversed_PCA = t(t(trainingPCA$x[, 1:numComponents] %*% t(trainingPCA$rotation[, 1:numComponents])) +
                   trainingPCA$scale + trainingPCA$center)
show_digit(reversed_PCA[4, ])


#6. Run kNN
accur_k <- matrix(ncol = 2, nrow = 8)
#we use a foreach loop to test values for k from 3 - 10 and then we append the result to our grid.
#This allows us the accuracy of multiple k values.
accur_k <- matrix(ncol = 2, nrow = 8)
i <- 1
for (k in 3:10){
  predict_k <-
    knn(
      train = final_Train,
      test = final_Test,
      cl = train_labels,
      k = k
    )
      accur_k[i, ] <- c(k, mean(predict_k == test_labels))
      i <- i + 1
}

#Review accuracy of model
accur_k
plot(accur_k)
#We can see that the model we chose was highly accurate.
#For a k of 5 we get 97.66% accuracy. This is the highest accuracy of our selected possibilities.


#7. Run NN (single layer perceptron)
#Format labels correctly for nnet function
NNetTrainLabels <- class.ind(train_labels)
NNetTestLabels <- class.ind(test_labels)
nnet <-
  nnet(final_Train,
       NNetTrainLabels,
       size = 20,
       softmax = TRUE)

#get predictions from model
pred = predict(nnet, final_Test, type = 'class')
cbind(head(pred), head(test_labels))

#Evaluate accuracy - 81% accuracy.
NNetEvaluation <- mean(pred == test_labels)
NNetEvaluation

#Summarise with confusion matrix as alternative
nnet_prediction <- predict(nnet, final_Test)
nnet_confusion_matrix <-
  confusionMatrix(nnet_prediction, test_labels)
sum(diag(nnet_confusion_matrix)) / 100  # Extracts accuracy from confusion matrix diagonal, in this case 81%

#Our original network gives us an 81% classification accuracy.
#To improve this we're going to use caret to tune the network
# Set up training parameters and a tuning grid to test multiple hidden node sizes.
caretTrainingParams <- trainControl(method = 'cv', number = 10)
NN_Tuning_Matrix <- expand.grid(.size = c(30, 40, 50), .decay = 0)
NN_Tuning_Matrix

#We run the nnet algorithm with our tuning matrix, so it tests all hidden node sizes that we have suggested
NNetVersion3 <- caret::train(
  final_Train,
  train_labels_factor,
  trControl = caretTrainingParams,
  method = 'nnet',
  tuneGrid = NN_Tuning_Matrix,
  MaxNWts = 30000
)

#Displays accuracy by k-fold retesting
NNetVersion3

#Alternatively test it against separate test dataset and produce a confusion matrix
NNet_Version3_Prediction <- predict(NNetVersion3, final_Test)
NNetV3_confusion_matrix <-
  confusionMatrix(NNet_Version3_Prediction, test_labels)
NNetV3_confusion_matrix
sum(diag(NNetV3_confusion_matrix)) / 100


#8. Try multi-layer perceptron for our neural network
#Clearly with only a single layer our accuracy will be compromised.
#We will use a multi-layer structure to improve accuracy with package RSNNS

caretTrainingParams <-
  trainControl(method = 'cv',
               number = 3,
               allowParallel = TRUE) #number possibly can't be 1

MLP_grid <- expand.grid(layer1 = 80,
                        layer2 = 60,
                        layer3 = 40)

MLP_1 <- caret::train(
  final_Train,
  train_labels_factor,
  trControl = caretTrainingParams,
  method = 'mlpML',
  tuneGrid = MLP_grid
)

#call it to see accuracy based on k-fold
MLP_1

#Alternatively assess accuracy against confusion matrix
MLP_1_Prediction <- predict(MLP_1, final_Test)
MLP_1_confusion_matrix <-
  confusionMatrix(MLP_1_Prediction, test_labels)
MLP_1_confusion_matrix
sum(diag(MLP_1_confusion_matrix)) / 100
