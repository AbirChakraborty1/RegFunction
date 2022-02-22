\name{rf_pt_reg}
\alias{rf_pt_reg}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Custom function for regression by Random Forest with Parameter Tuning
}
\description{
This function takes training data, tesing data, target variable, tuning parameters, and predict target variable of the training data by Random Forest model with Parameter Tuning.
}
\usage{
rf_pt_reg(train,test,target,
drop = NULL,num_trials=10,
mtry=c(1:7),
ntree=c(50,100,200,500,700),
maxnodes=c(5,10,15,20,30,50),
nodesize=c(1,2,5,10),
type='NULL'
)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{train}{
Training Data as a dataframe.
}
\item{test}{
Testing Data as a dataframe.
}
\item{target}{
Name of the variable that needs to predict
}
\item{drop}{
Name of the variable that needs to be dropped before building the model.
}
\item{num_trials}{
Total number of trails to find best  combination of Random Forest Tuning parameters causing minimum error
}
\item{mtry}{Random Forest uning parameter: number of predictors sampled for spliting at each node.Default is c(1:7)}

\item{ntree}{Random Forest Tuning parameter: number of trees grown.Default is c(50,100,200,500,700).
}
\item{maxnodes}{
Random Forest Tuning parameter: The size of the tree. Default is c(5,10,15,20,30,50)}

\item{nodesize}{
Random Forest Tuning parameter: Max number of terminal nodes.Default is c(1,2,5,10)
}
\item{type}{
if type = "tuning", it will return the least error with best value of the best parameter combination,any other value will return the prediction on testing data.
}
}
\value{
The value returns from this function is a numeric vector.
}
\author{
ABIR CHAKRABORTY < mail2abirchakraborty@gmail.com >
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
