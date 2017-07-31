package com.archimedon_lab.breeze

import breeze.linalg._
import java.io.File
import breeze.numerics._
import breeze.linalg.operators
import breeze.plot._
import breeze.optimize._
import breeze.stats._

object homework {

  def main(args: Array[String]): Unit = {
    runWeek1;
  }

  def runWeek1() = {
    val data = load("ex1data1.txt")
    runEx1_1(data)
    runEx1_2(data)
  }

  def runEx1_2(data: DenseMatrix[Double]) = {
    val m = data.rows;
    val n = data.cols;
    // Setup the data matrix appropriately, and add ones for the intercept term
    val X: DenseMatrix[Double] = DenseVector.horzcat(DenseVector.ones[Double](m), data(::, 0));
    val y = data(::, 1).asDenseMatrix.reshape(m, 1);
    printf("X with bias column size(%d x %d)\n", X.rows, X.cols);

    // Initialize fitting parameters
    val init_theta: DenseMatrix[Double] = DenseMatrix.zeros(n, 1);
    printf("init_theta size(%d x %d)\n", init_theta.rows, init_theta.cols);

    // Some gradient descent settings
    val iterations = 1500;
    val alpha = 0.01;

    printf("\nRunning Gradient Descent ...\n")
    val (theta, hist) = linearGradientDescent(X, y, init_theta, alpha, iterations);
    // print theta to screen
    printf("Theta found by gradient descent:\n %s\n", theta.toString());
    printf("Expected theta values (approx)\n");
    printf(" -3.6303\n  1.1664\n\n");

    val f = Figure();
    val p = f.subplot(0);
    p += plot(X(::, 1).toArray, y.toArray, '+');
    p.xlabel = "Population of City in 10,000s";
    p.ylabel = "Profit in $10,000s"
    p += plot(X(::, 1).toArray, (X * theta).toArray, '-');

    val fig2 = Figure();
    val pl = fig2.subplot(0);
    pl += plot((for (num <- 1 to hist.rows) yield num * 1d).toArray, hist(::, 0).toArray, '.');
    pl.xlabel = "Number of iterations";
    pl.ylabel = "J(θ_0, θ_1)"
    // nef.saveas("dump.png")
  }

  def featureNormalize(X: DenseMatrix[Double]) : DenseMatrix[Double] = {
    val stats = meanAndVariance(X.toArray);
    (X :- stats.mean) :* sqrt(stats.variance);
  } 

  def runEx1_1(data: DenseMatrix[Double]) = {
    val m = data.rows;
    val n = data.cols;

    // Setup the data matrix appropriately, and add ones for the intercept term
    val X: DenseMatrix[Double] = DenseVector.horzcat(DenseVector.ones[Double](m), data(::, 0));
    val y = data(::, 1).asDenseMatrix.reshape(m, 1);
    printf("X %d x %d\n", X.rows, X.cols);

    // Initialize fitting parameters
    val theta: DenseMatrix[Double] = DenseMatrix.zeros(n, 1);
    printf("theta %d x %d\n", theta.rows, theta.cols);
    val cost = computeSqECost(X, y, theta)
    printf("With theta = [0 ; 0]\nCost computed = %f\n", cost);
    printf("Expected cost value (approx) 32.07\n");

    // further testing of the cost function
    val J = computeSqECost(X, y, new DenseMatrix[Double](2, 1, Array(-1.0, 2.0)));
    printf("\nWith theta = [-1 ; 2]\nCost computed = %f\n", J);
    printf("Expected cost value (approx) 54.24\n");

  }

  // TODO - use LBFGS minimizer
  def runEx1_9(data: DenseMatrix[Double]) = {
    val lbfgs = new LBFGS[DenseVector[Double]](100,4)

    def optimizeThis(init: DenseVector[Double]) = {
      val f = new DiffFunction[DenseVector[Double]] {
        def calculate(x: DenseVector[Double]) = {
          (norm((x - 3.0) ^:^ 2.0, 1), (x *:* 2.0) - 6.0)
        }
      }

      val result = lbfgs.minimize(f,init) 
      norm(result - 3.0,2) < 1E-10
    }
  }

  // θ_j := θ_j - α * ∑( ( hø(x[i]) - y[i] ) * x[i]_j );
  def linearGradientDescent(X: DenseMatrix[Double], y: DenseMatrix[Double],
                            theta: DenseMatrix[Double], alpha: Double, num_iters: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val m = X.rows;
    val coef = (1f / m) * alpha
    val J_history: DenseMatrix[Double] = DenseMatrix.zeros(num_iters, 1);

    // restheta.size(2 x 1)
    var newtheta = theta;
    for (i <- 0 to num_iters - 1) {  // TODO - make recursive... for-loop creates extra objects
      //  hø(x[i]).size(m x 1)
      val hypEq = (X * newtheta);
      // (hø(x[i]) - y[i] ).size(m x 1)
      val errorDiff = hypEq - y;
      val sumOfDiffOXDis = (errorDiff.t * X).t;
      val delta = sumOfDiffOXDis :* coef;
      newtheta = newtheta - delta;
      // Save the cost J in every iteration    
      J_history(i, ::) := computeSqECost(X, y, newtheta);
    }
    (newtheta, J_history)
  }

  def load(filepath: String): DenseMatrix[Double] = {
    val fileUrl = getClass.getResource(filepath)
    csvread(new File(fileUrl.getPath), ',')
  }

  def computeSqECost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double]): Double = {
    val hypform = ((X * theta) - y).toDenseVector
    val sumOfSqrDiffs = hypform.t * hypform
    sumOfSqrDiffs / (2 * X.rows)
  }

  def computeVecCost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double]): Double = {
		  val hypform = ((X * theta) - y).toDenseVector
				  val sumOfSqrDiffs = hypform.t * hypform
				  sumOfSqrDiffs / (2 * X.rows)
  }

  def plotScatter(x: DenseMatrix[Double], y: DenseMatrix[Double]): (Figure, Plot) = {
    val f = Figure();
    val p = f.subplot(0);
    p += plot(x(::, 1).toArray, y.toArray, '+');
    (f, p)
  }

}
