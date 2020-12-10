# Trojan-Sleuth
## Background
The Trojan-Sleuth project is a competing method in IARPA's TrojAI competition. The goal of this competition is to accurately assess whether a pre-trained model has been infected with a trojan during its training process. The current iteration only deals with the image domain.
## Intuition For Our Method
A trojan-infected model, in the context of a neural network, is one that has been trained such that an impanted trigger in a sample will cause a misclassifiacation to a specified class. The Trojan-Sleuth method attempts to exploit this behavior by constructing a quasi-trigger, observing misclassification statistics, and using these statistics to predict whether the model has been trojaned.
## Methodology
The Trojan-Sleuth method is broken down into 3 files:
* statistic_generation.py - Produce misclassification statistics from training data (saves them as classification_stats.csv)
* regression_weights.py - Train a binary classifier on the statistics (reads data from classification_stats.csv)
* trojan_detector.py - Take in a model with example images, output a probability of the model being trojan
### Statistic Generation (statistic_generation.py)
#### Preprocessing
With each model in the NIST training data, we are given a set of example images drawn from the same distribuion from which the training images were drawn. The data can be found at https://pages.nist.gov/trojai/docs/data.html#round-3 and our code expects it saved in a folder called round3-dataset. We pre-process these images by first normalizing and formatting them to meet the requirements of the trained CNN. We then introduce an additional pre-processing step of blurring. We implement Gaussian blurring with a kernel size of 9 and sigma of 5 to remove extraneous detail from the image. The motivation for this is that the image's large dimensionality (256 x 256 x 3) may have introduced large amounts of irrelevant data hindering our analysis.
#### Quasi-trigger
We wish to simulate the misclassification behavior a trigger imposes on a trojan model by introducing a quasi-trigger. We obtain our quasi-trigger by running an image through the CNN, calculating the cross entropy loss, taking the gradient of the loss with respect to the image, taking the sign of this gradient, and multiplying it by a hyper-parameter epsilon. Through experimenation we have arrived on an epsilon of 0.3. This quasi-trigger has been observed to be effective in flipping the predicted class of example images.
#### Statistics
We are currently using 5 statistics in our method.
* Misclassificaion rate - This refers to the misclassification rate of all example images when their image-specific deltas are added
* MMC (Maximum misclassification concentration) - For each class, we add a common delta created from a represetative in the class, and observe the classifications this delta produces. We define misclassification concentration as the frequency of the most common incorrect classification divided by the total number of images in that class. We currently use statistics from 5 representatives and average the misclassification concentrations. MMC refers to the maximum of these across all classes. We expect the triggered class to have a very high MMC.
* Image-specific delta maximum misclassification concentration - The same as above except when image-specific deltas are used
* 1Q-MMC - From the clas with the maximum misclassification concentration, instead of taking the average misclassification concentration across the 5 representatives, we take the 1st quartile. This may be more robust to outliers
* 3Q-MMC - Same as above but with 3rd quartile
### Training the Classifier (regression_weights.py)
Once we have obtained statistics from our training data, we train a logistic regression model on it. We split into a training and test set to get a sense of its generalizability, and print some metrics. We then obtain the regression parameters.
### Evaluation (trojan_detector.py)
trojan_detector.py pre-processes data, obtains deltas, and computes statistics as was done in statistic_generation.py. It then used the parameters from the trained classifier to produce a probability of the model being trojan. This is the file submitted to the NIST server for evaluation. When it was submitted and evaluated on a set of 288 models, it produced a ROC-AUC of 0.7138 and cross entropy of 0.6207.
### Discussion
When looking at our results split by trigger-type (polygon vs instagram filter), we noticed that while our method performs reasonably well on polygon triggers (test cross-entropy of 0.52) it failed on instagram filter triggers (test cross-entropy of 0.73). We are therefore now moving in a direction to deal with instagram-triggered models by building a new type of quasi-trigger that expolits the different image-wide effects you would observe in an instagram filter. In regards to polygon triggers, we hope to improve results by building smarter quasi-triggers. For example replacing our uniform deltas with a delta that targets certain pixels more than others.
### Dependencies
Our code was run on a system using the following software
* Python 3.6.12
* CUDA 10.2
* numpy 1.19.2
* torch 1.7.0
* scipy 1.5.3
* scikit-learn 0.23.2
* scikit-image 0.17.2

This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

 
