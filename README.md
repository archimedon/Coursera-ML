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

  - In supervised learning, the task of the learning algorithm is to determine a suitable model (`h`), given a training set of data `X`. The algorithm has to derive our hypothesis function (h), testing successive variants of `h` ( `h_theta(x[i])` ) for _accuracy_ ... by comparing the result of `h_theta(x[i])` with the known output, `y[i]`.

### IV. Cost Function

  - A measure of the accuracy of our hypothesis function: `h_theta(x[i])`
  - For a Linear Regression problem, we assume a linear function thus:
		```
		h_theta(x[i]) = theta_0 + theta_1x[i]
		```
	- The cost/degree of error of our hypothesis h_theta(x) is calculated by taking the average of the sum of the square of _errors_ - the difference between `h_theta(x[i])` and `y[i]` (our known output for the value `x`)
	- Thus the cost function for Linear Regression, summarized as, `J(θ_0, θ_1)`, is calculated as...: ( where `m` is the number of samples and `i` an index into each row )
	
	```octave
	 J(θ_0, θ_1) = 1/2m * sum(  ( h_theta(x[i]) - y[i] )^2  )  ; for all [i]: 1 to m
	```
> This is called the "squared error function" or "squared error cost function". Other cost functions exists but SQE is good for linear regression problems. The goal is to minimize `J(θ_0, θ_1)`. Thus, if `J(θ_0, θ_1)=0` then our hypothesis function is an exact fit for our data.


____

##### Observations

- In `h_theta(x[i]) = theta_0 + theta_1 * x[i]`, note :
	
  - `theta_0` 		: represents the y-offset and
  - `theta_1`		: the slope of the line
  - If `theta_1` is negative, the slope of the tangent will be negative
  - Plotting `J(theta_1)`, i.e. let `theta_0 = 0` :
  	- `plot(theta_1, J(theta_1))` we get a bowl shaped graph, with an obvious minima

  - Plotting `J(θ_0, θ_1)`, that is, using **both** theta_0 & theta_1, we get a 3-dimensional graph, which looks even more like a bowl and also has a single global minima
  - The cost function is always **positive** because. even if the result of `h_theta(x) - y` is **negative**,  once squared, the result is **positve**

____

### V. Gradient Descent For Linear Regression

Gradient descent is an iterative approach for discovering that hypothesis `h_theta(x[i])` with the lowest cost, `J(θ_0, θ_1)`. Specifically, a gradient descent algorithm:
 

1. Starts with some given value for `theta_0` & `theta_1` (These are usually initilized to 0).
2. Keeps changing `theta_0` & `theta_1` in a manner that reduces the cost function `J(θ_0, θ_1)`
3. Repeat until we _hopefully_ find a minimum

Stepping through the samples in the training set ( `m` samples), the alogorithm repeats a step summarized as:
```
θ_j := θ_j - α * d/dθ J(Θ_0, Θ_1)
```

Where:
  - `α` - is the learning rate
  - `d/dθ `  - is the partial derivative of the cost function J(Θ_0, Θ_1) with respect to θ.

Taking the derivative of the point on the curve gives us the slope of the tangent at that point.


This algorithm is repeated for all samples, i from 1:m.

Here we can see that, if the cost is high, theta is penalized more significantly than if the cost, `J(Θ_0, Θ_1)`, were closer to zero.

Observe also, that θ_j plays the role of offset, and that `alpha * cost` affects the slope of the line. This line is a tangent to the curve described by plotting plot(θ_0 , θ_1, J(θ_0, θ_1)) (actually a 3D convex graph). By gradually adjusting the values of theta, the tangent (or tangent-plane) can move in a positive or negative direction.

##### Determining alpha

The learning rate, α, should be determined via some analysis of the data. Alterntively, one can experiment with different learning rates and plot the rate (change per iteration) at which the cost progresses to zero. If the learning rate is too large, it is possible that gradient descent will not converge or diverge, if is is too small it wil take many iterations to find the minimum. The prof. recommends starting with a value like: 0.03, then, if that does not descend fast enough, try 0.1. Alternatively, go in the other direction and try 0.01, etc... - it helps to plot(num_iterations, J(θ)).


### VI. Setting up Octave


## Review

> ### VII. Linear Algebra
> ### VIII. Matrices and Vectors

Only noting useful strategies and outliers here...:

If 

```
	A =				B =
		| 1 | 1 |			| 2 | 4 | 6 |
	 	| 2 | 3 |			| 1 | 3 | 3 |
	 	| 5 | 8 |			
	] 
```

All operations 


____

____

## Summary

In brief, linear regression is an approach for determining an equation which (we hope) fits, or models, our sample data. This equation is called a hypothesis.
Gradient descent is an iterative approach for discovering that hypothesis which best fits the data. In GD, in each iteration, the parameter values to the hypothesis are adjusted by tiny amounts, the hypothesis equation is calculated (using the sample data as input), and the _hyp_result_ is compared against the known output. The _difference_ between the expected output and the _hyp_result_ is fedback into the GD algorithm to influence the direction that the parameters are adjusted in the next iteration.
The difference between the _hyp_result_ and the sample data is considered a measurement of *error*. The average of the errors is a measure of how close the hypothesis is to being able to accurately predict outcomes.
Typically we work with the average of sum of the square of errors and is referred to as the *cost*. Thus, the goal of GD, is to find that equation with the smallest cost.

----


## Week 2

### I. Features and Polynomial Regression
### II. Feature Normalization
### IV. Normal Equation

The Normal Equation, an algebraic approach for determining theta(s), an alternative to Gradient Descent. Useful for small sets of features (100 - 1000, maybe 10000) depending on the data size of the features and how the calculations are being performed. Faster than gradient descent.


theta = (X' X)^-1 * X'y


(suppositions)

Depending on the language and support for vectorization, eqnNorm would likely make use of cores, threads and proecess in parallel, but yeilds no control for distributing the load.


## Week 3

### Logistic Regression

#### Classification

Examples of classification problems include things like:

Email: Spam or not spam?
Transaction: Fraudulent or not?
Tumor: Malignant/Benign?

In these types of classifications (Binary classification), `y` can have 2 possible values: `0` (No) or `1` (Yes).

In Logistic Regression, we need a hypothesis function that returns values that classify the data in the sample. Our hypothesis function has to return a value between 0 and 1. 

```octave
	0 ≤ hø(x) ≤ 1
```

In LR, we reduced the definition of the hypothesis function to:

```
hø(x) = transpose(θ) * X
```

In Logistic Regression, our hypothesis function equals the sigmoid function `g(z)`:


```octave

	hø(x) = g( θ' * X )

	z = θ' * X

given:

	g(z) = 1 / (1 + e^-z)


then,

	hø(x) = 1 / (1 + e^-[(θ' * X)])
```


- By plugging into the Sigmoid function, hø(x) will be between 0 and 1 because the Sigmoid function is an asymtope between Y=0 and Y=1 that crosses the Y-axis at `0.5`.

- If hø(x) = 0.7 . We treat that result as "hø(x) estimates probability that y=1, is 0.7". Mathematically we express this as:

```
	hø(x)  = P(y = 1 | x;θ)

"The probability that y=1, given x, parameterized by theta"

x - is a feature(s) of the patient

- P(y = 0 | x;θ) + P(y = 1 | x;θ) = 1
- P(y = 0 | x;θ) = 1 - P(y = 1 | x;θ)

- **Decision Boundary**
  - When h_theta(x) ≥ 0.5 we predict y = 1 and 
  - When h_theta(x) < 0.5 we predict y = 0

Understanding by example"

```
Given:

	hø(x) = g( θ_0 + (θ_1 * x_1) + (θ_2 * x_2) )

then

	z = θ_0 + (θ_1 * x_1) + (θ_2 * x_2)

If theta were set to:

	[	-3
 = 		1
 		1
 	]
```
We would predict "y = 1" if:

```
 	z ≥ 0

(plugging in theta) we get ..:

	-3	+ x_1 + x_2 ≥ 0

  = x_1	+ x_2 ≥ 3

If we assume an equals instead of a less-than-or-equal, then we're looking at the equation of a straight line:

	x_1	+ x_2 = 3

Which is simple. X_1 == 0 when x_2 == 3 and vice versa.

Thus the region where our hypothesis will predict `y = 1` will be that everything where  X_1 ≥ 3 while x_2 ≥ 3

Image("decision boundary- ex1")

Image("non-linear decision boundary") - A circle
given:

theta =


	[	-1
		0
		0
 = 		1
 		1
 	]

g( θ_0 + (θ_1 * x_1) + (θ_2 * x_2) + (θ_3 * x_3^2) + (θ_4 * x_4^2) )


```

### Cost Function (reloaded)

In Linear regression we had:

```
	J(Ø) =  1⁄m ∑ 1/2 ( hø(x[¡]) - y[¡] )^2

```

To make the transition, lets reduce the cost element to a function by replaing the squared-errors component:

```
	J(Ø) =  1⁄m ∑ Cost( hø(x[¡]), y[¡] )


This exposes the fact that the cost, J(Ø), is the sum of the individual errors between x[¡] & y[¡].

Thus we can also say:

	Cost( hø(x[¡]), y[¡]) = 1/2 ( hø(x) - y )^2


This can be interpreted as: the cost to pay is "1/2 the squared error".


	1/2 ( hø(x) - y )^2

Plugging a sigmoid function into 

	1/2 ( g(Ø' * x) - y )^2

Would yeild a non-convex curve; many local minimums.


#### A Cost() function for Logistic Regression
https://www.coursera.org/learn/machine-learning/supplement/bgEt4/cost-function

									-log( hø(x) )		if y = 1
Cost( hø(x), y) = BOTH {
									-log( 1 - hø(x) )	if y = 1


Cost = 0 if y = 1, hø(x) = 1
But as 		hø(x) --> 0
			Cost  --> infinity

-log( 1 - z ) starts from 0,0 and tends to infinity as hø(x) approaches 1


A compact form that compresses the 2 equesions (above) into one:

	Cost(hø(x), y) = -y log(hø(x)) - (1 - y ) (log(1 - hø(x)))

Therefore the Cost function for logistic regression is:

		J(Ø) =  1⁄m ∑ Cost( hø(x[¡]), y[¡] )

	=	- 1⁄m ∑ y[¡] * log(hø(x[¡])) + ( 1 - y[¡] ) (log(1 - hø(x[¡])))


To fit paramters Ø:

	To minimize J(Ø)

We go back to the GD algorithm:


	θ_j := θ_j - α * ∂/∂θ_j J(Θ)

Taking the derivative

	θ_j := θ_j - α * ∑( ( hø(x[i]) - y[i] ) * x[i]_j );


To make a prediction given a new x recall, the logistic hypothesis is wrapped in a sigmoid function:


	hø(x) = 1 / (1 + e^-[(θ' * X)])


A vectorized form for shifting theta:

 θ := θ − α⁄m * X' * ( g(X * θ) − y⃗ )


## AdvancedOptimzation Concepts

The gradient algorithm, can be decomposed into 2 parts:
-  J(Ø)				: 	a Cost() function
-  ∂/∂θ J(Θ)		: 	a function to compute the gradient - the slope defined by the derivative

There are other optimization algorithms, some alredy built-into Octave:
- "Conjugate gradient", "BFGS", and "L-BFGS"



## Multiclass classification

To classify into multiple buckets for example, directing mail to folders:
	
	Work, Friends, Social, Spam

One strategy is the **One vs All** strategy. The steps are:

1. Train a logistic regression **classifier** `hθ(x)` for each class￼_i_ to predict the probability that ￼ ￼y = i

2. select the classifier with `max(hθ(x))` 



That is for each class ¡:
```

hø[¡](x) = P(y = ¡ | x; Ø) (¡ = 1,2,3)

```

## Regularization

Choose an equation that fits the data....

We've been using a linear hypothesis function to fit the data, assuming, the larger the house the more it will sell for.. despite the fact that we see that prices plateau beyond a certain size.

Here we can say that a linear equation has "high-bias". As though the equation has a strong preconception about the data. This is called over-fitting

On the other extreme, if we have a 4th-order polynomial function that touches nearly every point. While it fits the data, it probably won't generalize enough to make an accurate prediction. "high-variance"/"under-fitting"

Given a data spread that grows then plateaus, a qudratic function may be the best fit and most generalized.

- if for matrix X, m ≤ n, X will be non-invertible


## Determine the degree of polynomial that will fit, 

#### Addressing Overfitting

- If we have a lot of feature and too little data, samples ( n versus m)


Options:

1) Reduce the number of features:
  - Manually select which features to keep.
  - Use a model selection algorithm (studied later in the course).

2) [Regularization](https://www.coursera.org/learn/machine-learning/supplement/1tJlY/cost-function)
  - Keep all the features, but reduce the magnitude of parameters θj.
  - Regularization works well when we have a lot of slightly useful features.

(from https://www.coursera.org/learn/machine-learning/supplement/VTe37/the-problem-of-overfitting )


