##
## R code to train the causal random forest
##

#library(grf)  # Some problem with R 4.0
library(uplift)

load_training_data = function(filename, path="./tmp/"){
	# Function for loading data stored in python process.
	X = read.csv(paste(path, filename, '.train_X.csv', sep=""), header=FALSE)
	y = read.csv(paste(path, filename, '.train_y.csv', sep=""), header=FALSE)
	t = read.csv(paste(path, filename, '.train_t.csv', sep=""), header=FALSE)
	return(list(X=X, y=unlist(y), t=unlist(t)))
}

load_data = function(filename, path='./tmp/', set='test'){
	  # Function for loading validation and testing data
	  # (only X).
	  #
	  # Args: 
	  # set (str): {'test', 'val'}
	  X = read.csv(paste(path, filename, '.', set, '.csv', sep=""), header=FALSE)
	  return(X)
}

save_data = function(item, filename, path='./tmp/'){
	  # Function for storing predictions
	  write.table(item, paste(path, filename, '.predictions.csv', sep=""),
	  row.names=FALSE, col.names=FALSE)
}

train_model = function(X, y, t){
	    # Function for training causal forest
	    model = causal_forest(X, y, t)
	    return(model)
}

save_model = function(model, filename, path='./tmp/'){
	   # Serialize the model and save to disk
	   saveRDS(model, file=paste(path, filename, ".model.rds", sep=""))
}

load_model = function(filename, path='./tmp/'){
	   # Load the serialized model from disk
	   model = readRDS(paste(path, filename, ".model.rds", sep=""))
	   return(model)
}

train_crf = function(filename, path='./tmp/', k=1){
      # Function to encapsulate everything needed in the training
      # phase.
      # Args:
      # k (int): k for k undersampling. Does not change data, only
      # stores the model with different name.
      print("Loading data...", end=" ")
      data = load_training_data(filename, path)
      print("Done.")
      print("Training causal random forest model...", end=" ")
      model = train_model(data$X, data$y, data$t)
      print("Done.")
      print("Saving model...", end=" ")
      save_model(model, paste(filename, k, sep=""), path=path)
      print("Done.")
}

uplift_rf_model = function(X, y, t){
  # Function for training uplift RF (Guelman & al. 2015)
  # Where do I do the cross-validation?
  model = upliftRF(x=X, y=y, ct=t,
                   # Guelman & al. used [1:length(colnames(X))]
                   mtry = 5,  # number of variables to try for each tree ("random subset")
                   # Guelman & al. used [100 * i for i in range(20)]
                   n_tree = 3,  # number of trees in forest.
                   split_method = 'KL',  # can also be 'ED', 'Chisq', 'L1', 'Int'
                   minsplit=3,  # minsplit - 1 is the stopping criterion (no more splits)
                   verbose=TRUE)
  return(model)
}

predict_crf = function(filename, path='./tmp/', k=1, set='test', load_model=TRUE){
	# Function to encapsulate everything needed in the prediction
	# phase.
	# Args:
	# k (int): k for k-undersampling. Used to find right model.
	# '1' indicates that no undersampling was done (default).
	# set (character): {'test', 'val'} Used to find right dataset.

	# Load data from disk:
	print("Loading data from disk...", end=" ")
	data = load_data(filename, path, set=set)  # This loads only X.
	print("Done.")
	if( load_model ) {
	    # Load model from disk:
	    print("Loading model from disk...", end=" ")
	    model = load_model(paste(filename, k, sep=""), path)
	    print("Done.")
	}
	# Use model to make predictions:
	print("Predicting...", end=" ")
	predictions = predict(model, data)
	print("Done.")
	# Store predictions to disk:
	print("Storing predictions to disk...", end=" ")
	save_data(predictions, filename=paste(filename, '.', k, sep=""),
	path=path)
	print("Done.")
}


train_uplift_rf = function(filename, path='./tmp/', k=1){  # no 'k' parameter? (filename!)
  # Function to encapsulate training of
  # uplift random forest (name? Guelman & al. 2015)
  print("Loading data...", end=" ")
  data = load_training_data(filename, path)
  print("Done.")
  # What format is the data in now?
  print("Training uplift random forest...", end=' ')
  model = upliftRF(x=data$X, y=data$y, ct=data$t,
                   split_method='KL')
  # Other parameters as default. 
  # Should decide what I want this R-function to do.
  # Cross-validation? Parameter selection?
  # If those parts are done on the python-side, then
  # loading of files should be done only once (outside of
  # the training function).
  print("Done.")
  print("Saving model...", end=" ")
  save_model(model, paste(filename, k, sep=""), path=path)
  print("Done.")
  # Maybe return model so that it does not need to be loaded
  # separately?
  return(model)
}


predict_uplift_rf = function(filename, path='./tmp/', k=1, set='test', load_model=TRUE){
  # Function to encapsulate everything needed in the prediction
  # phase.
  # Args:
  # k (int): k for k-undersampling. Used to find right model.
  # '1' indicates that no undersampling was done (default).
  # set (character): {'test', 'val'} Used to find right dataset.
  
  # Load data from disk:
  print("Loading data from disk...", end=" ")
  data = load_data(filename, path, set=set)  # This loads only X.
  print("Done.")
  if( load_model ) {
    # Load model from disk:
    print("Loading model from disk...", end=" ")
    model = load_model(paste(filename, k, sep=""), path)
    print("Done.")
  }
  # Use model to make predictions:
  print("Predicting...", end=" ")
  predictions = predict(model, data)
  print("Done.")
  # Store predictions to disk:
  print("Storing predictions to disk...", end=" ")
  save_data(predictions, filename=paste(filename, '.', k, sep=""),
            path=path)
  print("Done.")
}

