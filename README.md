
# Machine Learning Engineer Portfolio
This is a collection of fully operational machine learning projects developed as part of the Udacity's *[Machine Learning Engineer Nanodegree Program](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t#)*, the *[Deep Learning Nanodegree Program](https://www.udacity.com/course/deep-learning-nanodegree--nd101)* and the LinkedIn Learning *[Advance Your Skills in Deep Learning and Neural Networks](https://www.linkedin.com/learning/paths/advance-your-skills-in-deep-learning-and-neural-networks)* learning path.


## Deep Learning

### Price Prediction for ecommerce products  
> **Keywords**: Deep Learning | Price Prediction | CNN | Neural Networks | Data exploration. *[View Report](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/deep-learning/ml-price-prediction/Capstone%20Report.pdf)*, *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/deep-learning/ml-price-prediction/capstone_project.ipynb)*.

Using the official application of an e-commerce site, a historical sales database corresponding to a product category has been produced. By using this dataset containing all the information related to the products and their corresponding price, it is possible to train an algorithm capable of predicting the optimal competitive price for the publication of a new product. The motivation for pursuing this project is due to the fact that e-commerce companies require a holistic understanding of prices, and they must establish it intelligently as a marketing weapon.

### DCGAN: Deep Convolutional Generative Adversarial Networks
> **Keywords**: DCGAN | Adversial Networks | Convolutional | Deep Learning | Discount Factor. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/deep-learning/dcgan-image-generator/DCGAN.ipynb)*

In this notebook, a GAN using convolutional layers in the generator and discriminator is bulilt to generate numbers. This is called a Deep Convolutional GAN, or DCGAN for short. The DCGAN architecture was first explored last year and has seen impressive results in generating new images

### Face Generator
> **Keywords**: DCGAN | Adversial Networks | Convolutional | Deep Learning | Discount Factor. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/deep-learning/face-generation/dlnd_face_generation.ipynb)*

In this notebook,a generative adversarial network is built to generate new images of faces. This is called a Deep Convolutional GAN, or DCGAN for short. The DCGAN architecture was first explored last year and has seen impressive results in generating new images

### Image Classification
> **Keywords**: Deep Learning | CNN | Convolutional Neural Network | Image Classification | ResNet. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/deep-learning/dog-image-classifier/dog_app.ipynb)*

Classified images from the CIFAR-10 dataset. The dataset consisted of airplanes, dogs, cats, and other objects. The project involed preprocessing the images, then training a convolutional neural network on all the samples. The images needed to be normalized and the labels had to be one-hot encoded. Built convolutional, max pooling, dropout, and fully connected layers and see the neural network's predictions on the sample images.


## Reinforcement Learning

### Train a Quadcopter How to Fly
> **Keywords**: Reinforcement Learning | Deep Deterministic Policy Gradients | DDPG | Actor-Critic| Experience replay. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/reinforcement-learning/rl-quadcopter/Quadcopter_Project.ipynb)*

Design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm. The algorithm used was a Deep Deterministic Policy Gradients (DDPG) as described in the Continuous control with deep reinforcement learning paper. The choice was based on the requirement to support learning in a continuous action space.

![Quadcopter fly](https://raw.githubusercontent.com/nitsuga1986/machine-learning-nd-portfolio/master/reinforcement-learning/rl-quadcopter/simulation/flight_2.gif)

### Deep Q-learning - Cart Pole
> **Keywords**: Reinforcement Learning | Q-Learning | Cart Pole | OpenAI Gym | Experience replay. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/reinforcement-learning/Q-learning-cart/Q-learning-cart.ipynb)*

Build a neural network that can learn to play games through reinforcement learning. More specifically, we'll use QQ-learning to train an agent to play a game called Cart-Pole. In this game, a freely swinging pole is attached to a cart. The cart can move to the left and right, and the goal is to keep the pole upright as long as possible.

### Train a Self Driving Car
> **Keywords**: Reinforcement Learning | Q-Learning | Driving Agent | Optimal Policy | Discount Factor. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/reinforcement-learning/self-driving-car/smartcab.ipynb)*

Applied reinforcement learning to build a simulated vehicle navigation agent. This project involved modeling a complex control problem in terms of limited available inputs, and designing a scheme to automatically learn an optimal driving strategy based on rewards and penalties.

![Smart Cab](https://raw.githubusercontent.com/nitsuga1986/machine-learning-nd-portfolio/master/reinforcement-learning/self-driving-car/screenshots/training_trial_1.gif)

## Supervised Learning

### Finding Donors for CharityML
> **Keywords**: Supervised Learning | Classifiers | Feature Importance | Random forest | KNeighbors. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/supervised-learning/finding-donors/finding_donors.ipynb)*

Investigated factors that affect the likelihood of charity donations being made based on real census data. Developed a naive classifier to compare testing results to. Trained and tested several supervised machine learning models on preprocessed census data to predict the likelihood of donations. Selected the best model based on accuracy, a modified F-scoring metric, and algorithm efficiency.


## Unsupervised Learning

### Creating Customer Segments
> **Keywords**: Unsupervised Learning |  Clustering | PCA | Feature Transformation | Outlier Detection. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/unsupervised-learning/customer-segments/customer_segments.ipynb)*

Reviewed unstructured data to understand the patterns and natural categories that the data fits into. Used multiple algorithms and both empirically and theoretically compared and contrasted their results. Made predictions about the natural categories of multiple types in a dataset, then checked these predictions against the result of unsupervised analysis.


## Foundations & Metrics 

### Natural Language Processing - Word Embeddings  
> **Keywords**: Deep Learning | Word2vec | Skip-Gram | Neural Networks | NLP. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/foundations-metrics/word-embeddings/Skip-Gram_word2vec.ipynb)*.

Using TensorFlow to implement the word2vec algorithm using the skip-gram architecture. By implementing this, we'll explore about embedding words for use in natural language processing. This will come in handy when dealing with things like machine translation.

### Coding a Neural Network
> **Keywords**: Neural Network, Forward Propagation, Backpropagation, Gradient Descent, Tensorflow. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/foundations-metrics/coding-a-neural-network/coding_a_neural_network.ipynb)*

Build a neural network from scratch to carry out a prediction problem on a real dataset. Building a neural network from the ground up to have a better understanding of gradient descent, backpropagation, and other concepts that are important to know before we move to higher level tools. The data comes from the UCI Machine Learning Database.

### Predicting Boston Housing Prices
> **Keywords**: Data science | DecisionTreeRegressor | sklearn | Grid Search | Cross-Validation. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/foundations-metrics/predict-house-prices/boston_housing.ipynb)*

Built a model to predict the value of a given house in the Boston real estate market using various statistical analysis tools. Identified the best price that a client can sell their house utilizing machine learning.

### Titanic Survival Exploration 
> **Keywords**: Data science | Data exploration | NumPy | Pandas | matplotlib | scikit-learn. *[View Source](https://github.com/nitsuga1986/machine-learning-nd-portfolio/blob/master/foundations-metrics/titanic-survival-exploration/titanic_survival_exploration.ipynb)*

Create decision functions that attempt to predict survival outcomes from the 1912 Titanic disaster based on each passenger’s features, such as sex and age. Starting with a simple algorithm and increase its complexity until to accurately predict the outcomes for at least 80% of the passengers in the provided data.




--

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
