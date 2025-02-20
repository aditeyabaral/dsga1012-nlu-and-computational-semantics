# HW 1: Intrinsic Evaluation of Word Embeddings
**Due: February 27, 5:00 PM**

For this assignment, please complete the _problem set_ found in `hw1-pset.ipynb`. The problem set includes coding problems as well as written problems.

For the coding problems, you will implement functions defined in the Python files `embeddings.py` and `test_analogies.py`, replacing the existing code (which raises a `NotImplementedError`) with your own code. **Please write all code within the relevant function definitions**; failure to do this may break the rest of your code.

For the written problems, please submit your answers in PDF format, using the filename `hw1-written.pdf`. Make sure to clearly mark each problem in order to minimize the chances of grading errors.

You do not need to submit anything for Problems 1a, 2a, or 3a.

You are free to use any resources to help you with this assignment, including books, websites, or AI assistants such as ChatGPT (as long as such assistants are fully text-based). You are also free to collaborate with any other student or students in this class. However, you must write and submit your own answers to the problems, and merely copying another student's answer will be considered cheating on the part of both students. If you choose to collaborate, please list your collaborators at the top of your `hw1-written.pdf` file. If you choose to use ChatGPT or a similar AI assistant to help you with this assignment in any way, please include a file called `hw1-chatgpt-logs.txt` with a full transcript of all prompts and responses used for this assignment. Your use of AI assistants cannot involve generating images or any other content that cannot be included in a `.txt` file.

## Setup

You will need to complete your code problems in Python 3, preferably Python 3.8 or later. Apart from the standard Python libraries, the only dependency required for this assignment is [NumPy](https://numpy.org/).

## Submission

For your submission, please upload the following files to [Gradescope](https://www.gradescope.com):
* `embeddings.py`
* `test_analogies.py`
* `hw1-written.pdf`
* `hw1-chatgpt-logs.txt` (if you used ChatGPT or a similar text-based AI assistant)

Do not change the names of these files, and do not upload any other files to Gradescope. Failure to follow the submission instructions may result in a penalty of up to 5 points.

## Grading

The point values for each problem are given below. Problem 1c is worth 5 extra credit points, but the maximum possible grade on this assignment is 100 points. If you earn a total of 100 points or more _including the extra credit points_, then your grade will be 100.

| Problem | Problem Type | Points |
|---|---|---|
| 1b: Implement the Embeddings Class | Code | 20 |
| 1c: Extra Credit | Written | 5 EC |
| 2b: Implement Data Loading Script | Code | 20 |
| 3b: Calculate Cosine Similarity | Code | 10 |
| 3c: Find Neighboring Words | Code | 10 |
| 3d: Write Testing Script | Code | 20 |
| 4a: Syntactic vs. Semantic Relation Types | Written | 10 |
| 4b: Effect of Lenience | Written | 5 |
| 4c: Qualitative Evaluation | Written | 5 |
| **Total** | | **100** |

### Rubric for Code Problems
Code questions will be graded using a series of [Python unit tests](https://realpython.com/python-testing/). Each function you implement will be tested on a number of randomly generated inputs, which will match the specifications described in the function docstrings. **The unit tests will run immediately upon submitting your solution to [Gradescope](https://www.gradescope.com), and you will be able to see the results as soon as the tests have finished running.** Therefore, you are encouraged to debug and resubmit your code if one or more unit tests fail. 

For code questions, you will receive:
* full points if your code runs and passes all test cases
* at least 5 points if your code runs but fails at least one test case
* 0 points if your code does not run.

Partial credit may be awarded at the graders' discretion, depending on the correctness of your logic and the severity of bugs or other mistakes in your code. All code problems will be graded **as if all other code problems had been answered correctly**. Therefore, an incorrect implementation of one function should (in theory) not affect your grade on other problems that depend on that function.

### Rubric for Written Problems
For written problems, you will receive:
* full points if results are reported accurately and are accompanied by at least 1 to 2 sentences of thoughtful 
  analysis 
* at least 2 points if a good-faith effort (according to the TAs' judgement) has been made to answer the question
* 0 points if your answer is blank.

Partial credit may be awarded at the TAs' discretion.

