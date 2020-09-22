1. trainLinearlySeparable.txt : 
    The first row contains three integers: the no. of features (d), the no. of classes (m) and the total number of samples (n).
    Each of the following n lines contains d+1 numbers separted by spaces and/or tabs where first d values are features and the last one is the class value.


2. trainLinearlyNonSeparable.txt:
   The file format is sample as trainLinearlySeparable.txt; however, the samples are not linearly separable.
   This file can be used to train the pocket algorithm.

3. testLinearlySeparable.txt :
   The file consists of multiple line of numbers.
   Each line represents a sample consisting of d feature valuse and its true class.

4. testLinearlyNonSeparable.txt:
   The file has the same format as testLinearlySeparable.txt.
   However it should be used to verify the performance of pocket algorithm.