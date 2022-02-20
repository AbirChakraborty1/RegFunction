\name{lin_reg}
\alias{lin_reg}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Custom function for linear regression
}
\description{
This function takes training data, tesing data, target variable and predict target variable of the training data
}
\usage{
lin_reg(train, test, vif = 5, p = 0.05, target, drop = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{train}{
Training Data as a dataframe.
}
\item{test}{
Testing Data as a dataframe.
}
\item{vif}{
Cut off value of VIF
}
\item{p}{
Cut off of p-value
}
\item{target}{
Name of the variable that needs to predict
}
\item{drop}{
Name of the variable that needs to be dropped before building the model.
}
}
\details{
This is a custom function for linear regression, it builds a linear model first on the training data.

The argument 'drop' removes the variables not needed to build the model.

The argument 'vif' set the cut off of VIF and drop the variable with highest vif one by one till the maximum value of vif is less than cut-off.

The argument 'p' set the cut off of p-value and drop the variable with highest p-value one by one till the maximum value of p-value is less than cut-off.

The remaining variables are used to make the final linear model and predict the target variable on the training data.
}
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


