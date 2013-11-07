hacrf_sparse_cpp
================

Hidden alignment conditional random field for classifying string pairs.

Provides a command line interface to train and test HACRF model on example
string pairs and to label new pairs.

For more information on the model, please see 
*A Conditional Random Field for Discriminatively-trained Finite-state StringEdit Distance* 
by McCallum, Bellare, and Pereira, and the report
*Conditional Random Fields for Noisy text normalisation* by Dirko Coetsee.

Example
-------
Currently, two classes are supported. Examples for class 1 can be
found in pos.txt and examples for class 2 in neg.txt.

pos.txt looks like
```
#ur#|#your#
#dont#|#don't#
#yo#|#you#
#cont#|#continue#
#dat#|#that#
...
```
where each line contains two strings, separated by the pipe character.
Each line is an example of a string pair belonging to class 1, and in neg.txt
each line is a class 2 example.

When a new string pair is given, the model predicts whether that pair is
class 1 or class 2.
