# Machine Learning Course Notes

## Week 1 - Summary of concepts:

### I. What is ML?

> "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."

### II. UnSupervised Learning vs. Supervised Learning

  + Supervised
 	- Our sample data contains inputs with known output values. That is to say, in our sample we have already characterized the observed outputs. The sample data is used to _train_/discover our predictive algorithm.
 	+ Types of problems/algorithms:
 		- Regression:
 		  - prediction against a continuum
 		- Classification:
 		  - predict a discrete value output
 	- "Support Vector Machine ... will allow us to compute an infinite number of features"

  + Unsupervised
 	- We start with uncharacterized data, not informed are to what each data point means. The learning algorithm is tasked with finding some structure in the data.
 	+ Types of problems/algorithms:
 		+ Clustering:
 			- Grouping the data based, perhaps on degree of _similarity_ between features, or content
 		+ Non-clustering:
 			- Find a pattern in the data. Ex: The [Cocktail Party](https://en.wikipedia.org/wiki/Cocktail_party_effect) Problem

> Cocktail Party Problem
>```
> [W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');
>```


### III. Model Representation

  - In supervised learning, the task of the learning algorithm is to determine a suitable model (`h`), given a training set of data `X`. The algorithm has to derive our hypothesis function (h), testing successive variants of `h` ( `h_θ(x[i])` ) for _accuracy_ ... by comparing the result of `h_θ(x[i])` with the known output, `y[i]`.

### IV. Cost Function

  - A measure of the accuracy of our hypothesis function: `h_θ(x[i])`
  - For a Linear Regression problem, we assume a linear function thus:
		```
		h_θ(x[i]) = θ_0 + θ_1x[i]
		```
	- The cost/degree of error of our hypothesis h_theta(x) is calculated by taking the average of the sum of the square of _errors_ - the difference between `h_θ(x[i])` and `y[i]` (our known output for the value `x`)
	- Thus the cost function for Linear Regression, summarized as, `J(Θ_0 , θ_1)`, is calculated as...: ( where `m` is the number of samples and `i` an index into each row )
	
	```
	 J(Θ_0 , θ_1) = 1/2m * sum(  ( h_θ(x[i]) - y[i] )^2  )  ; for all [i]: 1 to m
	```
> This is called the "squared error function" or "squared error cost function". Other cost functions exists but SQE is good for linear regression problems. The goal is to minimize `J(Θ_0 , θ_1)`. Thus, if `J(Θ_0 , θ_1)=0` then our hypothesis function is an exact fit for our data.

____

##### Observations

- In `h_θ(x[i]) = θ_0 + θ_1x[i]`, note that:
	
  - `θ_0` 		: represents the y-offset and
  - `θ_1x[i]`	: the slope of the line
  - If `θ_1` is negative, the slope will be negative
  - Plotting `J(θ_1)`, i.e. let `Θ_0 = 0` :
  	- `plot(Θ_1, J(Θ_1))` we get a bowl shaped graph, with an obvious minima

  - Plotting `J(Θ_0 , θ_1)`, that is, using *both* theta_0 & theta_1, we get a 3-dimensional graph, which looks even more like a bowl and also has a single global minima.

____

### V. Gradient Descent For Linear Regression

Gradient descent (*batch gradient descent*) is an iterative approach for discovering that hypothesis `h_θ(x[i])` with the lowest cost, `J(Θ_0 , θ_1)`. Specifically, a gradient descent algorithm:

1. Takes some starting value for `Θ_0` & `θ_1` (These are usually initilized to 0).
2. Keep changing `Θ_0` & `θ_1` in a manner that reduces the cost function `J(Θ_0 , θ_1)`
3. Repeat until we _hopefully_ find a minimum


### VI. Univariate Linear Regression

##### Determining alpha

### IX. Setting up Octave


## Review

### VII. Linear Algebra

### VIII. Matrices and Vectors

____

## Summary

In brief, linear regression is an approach for determining an equation which (we hope) fits, or models, our sample data. This equation is called a hypothesis.
Gradient descent is an iterative approach for discovering that hypothesis which best fits the data. In GD, in each iteration, the parameter values to the hypothesis are adjusted by tiny amounts, the hypothesis equation is calculated (using the sample data as input), and the _hyp_result_ is compared against the known output. The _difference_ between the expected output and the _hyp_result_ is fedback into the GD algorithm to influence the direction that the parameters are adjusted in the next iteration.
The difference between the _hyp_result_ and the sample data is considered a measurement of *error*. The average of the errors is a measure of how close the hypothesis is to being able to accurately predict outcomes.
Typically we work with the average of sum of the square of errors and is referred to as the *cost*. Thus, the goal of GD, is to find that equation with the smallest cost.

----


## Week 2
#### Core concepts: Gradient Descent, Multivariate Linear Regression
## Synopsis

