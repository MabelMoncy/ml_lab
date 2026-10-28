# Experiment 6: Text Classification using Multinomial vs Bernoulli Naïve Bayes (20 Newsgroups Dataset)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load dataset
data = fetch_20newsgroups(
    subset='all',
    categories=['sci.space', 'rec.sport.baseball', 'comp.graphics'],
    remove=('headers', 'footers', 'quotes')
)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Multinomial and Bernoulli features
v1 = CountVectorizer(stop_words='english', max_features=5000)
v2 = CountVectorizer(stop_words='english', binary=True, max_features=5000)

X1 = v1.fit_transform(X_train)
T1 = v1.transform(X_test)
X2 = v2.fit_transform(X_train)
T2 = v2.transform(X_test)

# Train and evaluate models
for name, model, train, test in [
    ("Multinomial NB", MultinomialNB(), X1, T1),
    ("Bernoulli NB", BernoulliNB(), X2, T2)
]:
    model.fit(train, y_train)
    pred = model.predict(test)
    print("\n===", name, "===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average='macro'))
    print(classification_report(y_test, pred))
