# Machine Learning Basics

Machine learning is a branch of artificial intelligence in which a model
learns patterns from data instead of being programmed with explicit rules.
In supervised learning, the training data is labelled with the correct
answers, and the model learns to predict those labels for new inputs — for
example, classifying an email as spam or not spam. In unsupervised learning
the data has no labels, and the model instead finds structure, such as
grouping customers into clusters with similar behaviour.

Data scientists split their data into a training set, used to fit the model,
and a held-out test set, used to estimate how well the model generalises to
unseen data. This matters because of overfitting: a model that memorises the
training examples, including their noise, can perform very well on the
training set yet badly on new data. Techniques such as cross-validation,
regularisation and simply collecting more data help to reduce overfitting.

Modern machine learning systems often represent text, images and other
inputs as embeddings — dense numerical vectors in which similar items sit
close together. Embeddings power semantic search: a question and a passage
can be compared by the distance between their vectors, which is the idea
behind retrieval-augmented generation, where the most relevant passages are
fetched first and then handed to a language model so its answer is grounded
in real documents.
