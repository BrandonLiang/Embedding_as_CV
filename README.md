# Embedding as CV

This project aims to visualize BERT embeddings on a sample word after each epoch of training (Masked Language Modeling) and saves the embedding of each epoch as image for viewing.

This project uses [PyTorch Template](https://github.com/victoresque/pytorch-template) as the base template

## To Do
1. Fix save-best in python/base\_trainer.py like in [this tutorial](https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/)
2. create directory based on run/model name in config/<config.json> to store all artifacts: log, tensorboard log, model checkpoint, etc.
