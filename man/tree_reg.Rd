\name{tree_reg}
\alias{tree_reg}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Custom function for  regression by decision tree.
}
\description{
This function takes training data, tesing data, target variable and predict target variable of the training data}
\usage{
tree_func(train, test, target, drop = NULL)
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
Name of the variable where probability score needs to be predicted
}
\item{drop}{
Name of the variable that needs to be dropped before building the model.
}
}
\details{
This is a custom function for  regression by decision tree, it builds a tree model first on the training data.

The argument 'drop' removes the variables not needed to build the model.

The remaining variables are used to make the final tree model and predict the target variable on the training data.}
\value{
The value returns from this function is a numeric vector.

}
\references{
%% ~put references to the literature/web site here ~
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
