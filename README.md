# ts2v
This is an implemention of t2v with high dimentional temporal data.

The idea of embedding time to vector is from Time2Vec: Learning a Vector Representation of Time: https://arxiv.org/abs/1907.05321

We found the time embedding idea innotive and effective, thus applying it to our time series data. ts2v generates embeddings for a time series rather than a timestamp.

The results are tested under python3.8 and torch1.13.1.
