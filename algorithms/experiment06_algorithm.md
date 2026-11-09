# Experiment 6: Text Classification using Multinomial and Bernoulli Naive Bayes

## Algorithm

1. Import the required libraries for text processing, dataset splitting, Naive Bayes, and evaluation.
2. Load the 20 Newsgroups dataset with the categories `sci.space`, `rec.sport.baseball`, and `comp.graphics`.
3. Remove headers, footers, and quotes from the text data.
4. Split the text data into training and testing sets.
5. Create a CountVectorizer for word counts and another CountVectorizer with binary features.
6. Fit the first vectorizer on the training text and transform the training and testing text.
7. Fit the binary vectorizer on the training text and transform the training and testing text.
8. Create Multinomial Naive Bayes and Bernoulli Naive Bayes models.
9. Train each model using its corresponding training features.
10. Predict the classes for the test text using each model.
11. Print the accuracy, macro F1 score, and classification report for both models.
