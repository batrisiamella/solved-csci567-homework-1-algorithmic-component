Download Link: https://assignmentchef.com/product/solved-csci567-homework-1-algorithmic-component
<br>
<strong>Review </strong>In the lectures, we have described the least mean square (LMS) solution for linear regression as

<em>w</em>LMS = (<em>X</em>T<em>X</em>)−1<em>X</em>T<em>y                                                                                </em>(1)

where <em>X </em>is our design matrix (<em>N </em>rows, <em>D </em>+1 columns) and <em>y </em>is the <em>N</em>-dimensional column vector of the true values in the training data. We have assumed that each row of <em>X </em>has been appended with the constant 1 in Eqn. 1.

<strong>Question 1.1 </strong><em>X</em><strong><sup>T</sup></strong><em>X </em><strong>is not invertible              </strong>[Recommended maximum time spent: 15 mins]

In the lecture, we have described a practical challenge for linear regression. Specifically, the least square solution is not possible when <em>X</em><sup>T</sup><em>X </em>is not invertible. Please use a concise mathematical statement (<strong>in one sentence</strong>) to summarize the relationship between the training data <em>X </em>and the dimensionality of <em>w </em>when this bad scenario happens.

<em>What to submit: </em>your one sentence answer in the written report.

<strong>Question 1.2 Bias solution         </strong>[Recommended maximum time spent: 60 mins]

At the end of lecture 3, we mentioned that under certain assumption, the bias <em>b </em>has a solution being the mean of the samples

<em>,                                                                          </em>(2)

where <strong>1</strong><em><sub>N </sub></em>= [1<em>,</em>1<em>,…,</em>1]<sup>T </sup>is a <em>N</em>-dimensional all one column vector.

We can prove that it is true when <em>D </em>= 0, i.e. we ignore the features such that the design matrix is a column of 1’s, by the following procedure.

<table width="0">

 <tbody>

  <tr>

   <td width="264"><em>b</em><sup>∗ </sup>= argmin<em><sub>b </sub></em>k<em>y </em>− <em>b</em><strong>1</strong><em><sub>N</sub></em>k<sup>2</sup></td>

   <td width="241">Residual sum of squares</td>

   <td width="20">(3)</td>

  </tr>

  <tr>

   <td width="264"><strong>1</strong><sup>T</sup><em><sub>N</sub></em>(<em>y </em>− <em>b</em><sup>∗</sup><strong>1</strong><em><sub>N</sub></em>) = 0</td>

   <td width="241">Taking derivatives w.r.t <em>b</em></td>

   <td width="20">(4)</td>

  </tr>

 </tbody>

</table>

<em>y                                                                                                                                                </em>(5)

In this question, we would like you to generalize the proof above to arbitrary <em>D </em>and arrive at something similar to Eqn. 2.

Please follow the three-step recipe: 1) write out the residual sum of squares objective function; 2) take derivatives w.r.t. the variable you are interested in and set the gradient to 0; 3) solve for <em>b</em><sup>∗ </sup>and conclude. You will find out that you need one more condition to arrive at Eqn. 2, which is

(6)

This is to center the input data (excluding the appended constant) to be zero mean, which is a common preprocessing technique people use in practice. There is a simple explanation to Eqn. 6 — if the feature values are zero on average, the average response can only be caused by the bias (Eqn. 2).

<em>What to submit: </em>your fewer-than-10-line proof in the written report.

<h1>2           Logistic Regression</h1>

<strong>Review        </strong>Recall that the logistic regression model is defined as:

<em>p</em>(<em>y </em>= 1|<em>x</em>) = <em>σ</em>(<em>w</em><sup>T</sup><em>x </em>+ <em>b</em>)                                                                           (7)

(8)

Given a training set, where <em>y<sub>n </sub></em>∈ {0<em>,</em>1}, we will minimize the cross entropy error function to solve for <em>w </em>and <em>b</em>.

<table width="0">

 <tbody>

  <tr>

   <td width="531">minE(<em>w,b</em>) = min<sup>−X</sup>{<em>y<sub>n </sub></em>log[<em>p</em>(<em>y<sub>n </sub></em>= 1|<em>x<sub>n</sub></em>)] + (1 − <em>y<sub>n</sub></em>)log[<em>p</em>(<em>y<sub>n </sub></em>= 0|<em>x<sub>n</sub></em>)]} <em>w</em><em>,b   </em><em>w</em><em>,b</em><em>n</em></td>

   <td width="28">(9)</td>

  </tr>

  <tr>

   <td width="531">= min−<sup>X</sup>{<em>y<sub>n </sub></em>log<em>σ</em>(<em>w</em><sup>T</sup><em>x<sub>n </sub></em>+ <em>b</em>) + (1 − <em>y<sub>n</sub></em>)log[1 − <em>σ</em>(<em>w</em><sup>T</sup><em>x<sub>n </sub></em>+ <em>b</em>)]}</td>

   <td width="28">(10)</td>

  </tr>

 </tbody>

</table>

<em>w</em><em>,b n</em>

<strong>Question 2.1 Bias solution         </strong>[Recommended maximum time spent: 45 mins]

Consider if one does not have access to the feature <em>x </em>of the data, and is given a training set of

, where <em>y<sub>n </sub></em>∈ {0<em>,</em>1}. What would be the optimal logistic regression classifier in that

case? What is the probability that a test sample is labeled as 1? You could also compare your solution with Eqn. 2 in Question 1.2.

Specifically, please write out the objective function as in Eqn. 10. And solve for the optimal bias term <em>b</em><sup>∗</sup>.

<em>What to submit: </em>your fewer-than-5-line derivation and your formula for the optimal bias in the written report.

<strong>Programming component</strong>

<h1>3           Pipeline and Datasets</h1>

In this section, we will first explain the general coding pipeline you will use in Sect. 4, 5 and 6, and then describe the datasets.

<h2>3.1         Pipeline</h2>

A standard machine learning pipeline usually consists of three parts. 1) Load and preprocess the data. 2) Train a model on the training set and use the validation set to tune hyperparameters. 3) Test the final model and report the result. In this assignment, we will provide you a template for each section (linear regression.py, knn.py, and logistic.py), which follows this 3-step pipeline. We provide step 1’s data loading and preprocessing code, and define step 3’s output format to report the results. You will be asked to implement step 2’s algorithms to complete the pipeline. Do not make any modification to our implementations for step 1 and 3.

Please do not import packages that are not listed in the provided scripts. Follow the instructions in each section strictly to code up your solutions. <strong>DO NOT CHANGE THE OUTPUT</strong>

<strong>FORMAT</strong>. <strong>DO NOT MODIFY THE CODE UNLESS WE INSTRUCT YOU TO DO SO</strong>. A homework solution that does not match the provided setup, such as format, name, initializations, etc., <strong>will not </strong>be graded. It is your responsibility to make sure that your code runs with python3 on the VM.

Figure 1: Example output

<strong>Example output </strong>For linear regression.py in Sect. 4, you should be able to run it on VM and see output similar to Fig. 1.

<h2>3.2         Datasets</h2>

<strong>Regression Dataset </strong>The UCI Wine Quality dataset lists 11 chemical measurements of 4898 white wine samples as well as an overall quality per sample, as determined by wine connoisseurs. See <strong>winequality-white.csv</strong>. We split the data into training, validation and test sets in the preprocessing code. You will use linear regression to predict wine quality from the chemical measurement features.

<strong>Classification Datasets </strong>We describe three datasets that will be used for classification problems in this homework. Each dataset corresponds to a JSON file named as <strong>$dataset$.json</strong>. JSON is a lightweight data-interchange format, similar to a dictionary. After loading a dataset, you can access its training, validation, and test splits using the keys ‘train’, ‘valid’, and ‘test’, respectively. For example, suppose we load <strong>mnist subset.json </strong>to the variable <em>x</em>. Then, <em>x</em>[<sup>0</sup><em>train</em><sup>0</sup>] refers to the training set of <strong>mnist </strong><strong>subset</strong>. This set is a list with two elements: <em>x</em>[<sup>0</sup><em>train</em><sup>0</sup>][0] containing the features of size <em>N </em>(samples) ×<em>D </em>(dimension of features), and <em>x</em>[<sup>0</sup><em>train</em><sup>0</sup>][1] containing the corresponding labels of size <em>N</em>.

Next we will describe each one of the three datasets in more detail.

<ul>

 <li><strong>toydata1 </strong>includes 240 data instances with binary labels that are linearly separable. The data is split into a training set and a test set. You can look up training and test sets in <strong>json </strong>with [<sup>0</sup><em>train</em><sup>0</sup>] and [<sup>0</sup><em>test</em><sup>0</sup>], respectively.</li>

 <li><strong>toydata2 </strong>includes 240 data instances with binary labels that are not linearly separable. The data is split into a training set and a test set. You can look up training and test sets in <strong>json </strong>with [<sup>0</sup><em>train</em><sup>0</sup>] and [<sup>0</sup><em>test</em><sup>0</sup>], respectively.</li>

 <li><strong>mnist subset</strong>: MNIST is one of the most well-known datasets in computer vision, consisting of images of handwritten digits from 0 to 9. We will be working with a subset of the official</li>

</ul>

Figure 2: Left: <strong>toydata1</strong>, linearly separable; Right: <strong>toydata2</strong>, not linearly separable

version of MNIST. In particular, we randomly sample 700 images from each category and split them into training, validation, and test sets, which can be reached in <strong>mnist subset.json </strong>via keys [<sup>0</sup><em>train</em><sup>0</sup>], [<sup>0</sup><em>valid</em><sup>0</sup>] and [<sup>0</sup><em>test</em><sup>0</sup>], respectively.

<h1>4           Linear Regression</h1>

You are asked to implement 4 python functions for linear regression. The input and output of the functions are specified in linear regression.py. You should be able to run linear regression.py after you finish the implementation. Note that we have already appended the column of 1’s to the feature matrix, so that you do not need to modify the data yourself.

<strong>Question 4.1 Linear regression         </strong>Implement linear regression and return the model parameters.

<em>What to submit: </em>fill in the function linear regression noreg(X, y).

<strong>Question 4.2 Regularized linear regression </strong>To address the challenge described in Question 1.1, as well as other issues such as overfitting, we add regularization. For now, we will focus on <em>L</em><sub>2 </sub>regularization. In this case, the optimization problem is:

<em>w</em><em><sup>λ </sup></em>= argmin<em><sub>w</sub></em>                                                           (11)

where <em>λ </em>≥ 0 is a hyper-parameter used to control the complexity of the resulting model. When <em>λ </em>= 0, the model reduces to the usual (unregularized) linear regression. For <em>λ &gt; </em>0 the objective function balances between two terms: (1) the data-dependent quadratic loss function , and (2) a function of the model parameters.

Implement your regularized linear regression algorithm.

<em>What to submit: </em>fill in function regularized linear regression(X, y, <em>λ</em>).

<strong>Question 4.3 Tuning the regularization hyper-parameter </strong>Use the validation set to tune the regularization parameter <em>λ </em>∈ {0<em>,</em>10<sup>−4</sup><em>,</em>10<sup>−3</sup><em>,</em>10<sup>−2</sup><em>,</em>10<sup>−1</sup><em>,</em>1<em>,</em>10<em>,</em>10<sup>2</sup>}. We select the best one that results in the lowest mean square error on the validation set.

<em>What to submit: </em>fill in the function tune lambda(Xtrain, ytrain, Xval, yval, lambds).

<strong>Question 4.4 Mean square error </strong>Report the mean square error of the model on the given test set.

<em>What to submit: </em>fill in the function test error(w, X, y).

<h1>5           <em>k </em>Nearest Neighbors</h1>

<strong>Review </strong>In the lecture, we define the classification rule of the <em>k</em>-nearest neighbors (<em>k</em>NN) algorithm for an input example <em>x </em>as

<em>v<sub>c </sub></em>=               <sup>X </sup>1(<em>y<sub>i </sub></em>== <em>c</em>)<em>,</em>∀<em>c </em>∈ [<em>C</em>]                                                                                           (12)

<em>x<sub>i</sub></em>∈knn(<em>x</em>)

<em>y </em>= arg max<em>v<sub>c                                                                                                                                                                                          </sub></em>(13)

<em>c</em>∈[<em>C</em>]

where [<em>C</em>] is the set of classes.

A common distance measure between two samples <em>x<sub>i </sub></em>and <em>x<sub>j </sub></em>is the Euclidean distance:

<em>.                                                    </em>(14)

You are asked to implement 4 python functions for <em>k</em>NN. The input and output are specified in knn.py. You should be able to run knn.py after you finish the implementation.

<strong>Question 5.1 Distance calculation   </strong>Compute the distance between test data points in <em>X </em>and training data points in <em>X</em><sub>train </sub>based on Eqn. 14.

<em>What to submit: </em>fill in the function compute distances(Xtrain, X).

<strong>Question 5.2 </strong><em>k</em><strong>NN classifier </strong>Implement <em>k</em>NN classifier based on Eqn. 13. Your algorithm should output the predictions for the test set. <em>Important: </em>When there are ties among predicted classes, you should return the class with the smallest value. For example, when <em>k </em>= 5, if the labels of the 5 nearest neighbors happen to be 1<em>,</em>1<em>,</em>2<em>,</em>2<em>,</em>7, your prediction should be the digit 1.

<em>What to submit: </em>fill in the function predict labels(k, ytrain, dists).

<strong>Question 5.3 Report the accuracy            </strong>The classification accuracy is defined as:

# of correctly classified test examples

accuracy =                                             (15)

# of test examples

The accuracy value should be in the range of

<em>What to submit: </em>fill in the code for function compute accuracy(y, ypred).

<strong>Question 5.4 Tuning </strong><em>k </em>Find <em>k </em>among  that gives the best classification accuracy on the validation set.

<em>What to submit: </em>fill in the code for function find best k(K, ytrain, dists, yval).

<h1>6           Logistic Regression</h1>

You are asked to finish 3 python functions for logistic regression to solve the binary classification problem. The input and output are specified in logistic.py. You should be able to run logistic.py after you finish the implementation.

<strong>Question 6.1 Logistic regression classifier </strong>Find the optimal parameters for logistic regression using gradient descent. The objective is the cross-entropy error function described in class and in Eqn. 10.

We have initialized <em>w </em>and <em>b </em>to be all 0s (please do not modify this initialization procedure.). At each iteration, we compute the average gradients from all training samples and update the parameters using the chosen step size (or learning rate). We have also specified values for the step size and the maximum number of iterations (please do not modify those values). <em>What to submit: </em>fill in the function logistic train(Xtrain, ytrain, w, b, step size, max iterations).

<strong>Question 6.2 Test accuracy </strong>After you train your model, run your classifier on the test set and report the accuracy, which is defined as:

# of correctly classified test examples

# of test examples

<em>What to submit: </em>fill in the function logistic test(Xtest, ytest, w, b).

<strong>Question 6.3 Feature transformation toydata2 </strong>is not linearly separable and logistic regression does not perform well. Let’s try a simple modification to make it work better! Take element-wise square over the features of both [<sup>0</sup><em>train</em><sup>0</sup>] and [<sup>0</sup><em>test</em><sup>0</sup>] of <strong>toydata2</strong>. That is, each element in the feature matrix becomes the square of that element. Repeat the training process and report the final test accuracy. You should see the big difference between the two experiments.

<em>What to submit: </em>fill in the function feature square(Xtrain, Xtest).