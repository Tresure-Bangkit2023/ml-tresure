# Tresure Machine Learning Repository

Welcome to the Tresure Machine Learning Repository! This repository contains the machine learning models and code used in the Tresure app, a travel recommendation application that helps users discover affordable and exciting destinations.

## Overview

The Tresure app utilizes machine learning algorithms to provide personalized recommendations based on user preferences and location data. This repository houses the notebook used in the development of these recommendation systems.

## Model Overview

The Matrix Factorization Network (GMF) is a straightforward approach that utilizes point-wise matrix factorization to approximate the factorization of a matrix representing user-item interactions. In this network, Stochastic Gradient Descent (SGD) with Adam optimization is employed to factorize a (user x item) matrix into two matrices: one representing user latent features and the other representing item latent features.

The GMF network follows a simpler architecture compared to other models. It adopts the same setup as the Multilayer Perceptron (MLP) for embedding user and item features, followed by element-wise multiplication between the embeddings. The final output layer consists of a dense layer with a single neuron. Binary cross-entropy loss is used in conjunction with Adam optimization for training the network.

## Features

- Data preprocessing: Scripts and tools for cleaning and preprocessing the travel and user data before feeding it into the model.
- Recommendation models: This repository includes a machine learning model, collaborative filtering, implemented for the purpose of generating travel destination recommendations.
- Learning Rate Scheduler: Method for finding the best learning rate on the optimizer used.
- Evaluation and metrics: Tools for evaluating the performance of the recommendation models using appropriate metrics.

## Requirements

To run the notebook in this repository, you'll need the following dependencies:

- Python 3.x
- pandas
- numpy
- sklearn
- tensorflow
- matplotlib

## References
- https://medium.com/@victorkohler/collaborative-filtering-using-deep-neural-networks-in-tensorflow-96e5d41a39a1
