# Deep Learning History and Breakthroughs

## Early Foundations
The perceptron was introduced by Frank Rosenblatt in 1958 at the Cornell Aeronautical Laboratory. It was the first trainable neural network capable of binary classification. However, Minsky and Papert's 1969 book "Perceptrons" showed limitations of single-layer networks, leading to the first AI winter.

## Backpropagation Revolution
The backpropagation algorithm was popularized by Rumelhart, Hinton, and Williams in their 1986 paper "Learning representations by back-propagating errors." This allowed multi-layer networks to be trained effectively, though computational limits restricted applications for years.

## Convolutional Neural Networks
Yann LeCun developed LeNet-5 in 1998 for handwritten digit recognition (MNIST dataset). This architecture used convolutional layers, pooling, and fully connected layers. In 2012, Alex Krizhevsky won the ImageNet competition with AlexNet, achieving a top-5 error rate of 15.3% compared to the previous best of 26.2%. This marked the beginning of the deep learning revolution.

## Recurrent Neural Networks and LSTMs
Hochreiter and Schmidhuber introduced the Long Short-Term Memory (LSTM) architecture in 1997 to address the vanishing gradient problem in RNNs. LSTMs became the standard for sequence modeling tasks including machine translation and speech recognition until the transformer architecture emerged.

## The Transformer Architecture
The transformer was introduced by Vaswani et al. in the 2017 paper "Attention Is All You Need." This architecture replaced recurrence with self-attention mechanisms, enabling parallel processing and better handling of long-range dependencies. The transformer became the foundation for BERT (2018), GPT (2018), and all subsequent large language models.

## BERT and Pre-training
Devlin et al. introduced BERT in 2018, which used masked language modeling and next sentence prediction for pre-training. BERT achieved state-of-the-art results on 11 NLP tasks and established the pre-training then fine-tuning paradigm. BERT-large had 340 million parameters.

## GPT Models
OpenAI released GPT-1 in June 2018 with 117 million parameters using only the decoder portion of the transformer. GPT-2 was released in February 2019 with 1.5 billion parameters and demonstrated impressive text generation capabilities. GPT-3 was announced in June 2020 with 175 billion parameters, showing remarkable few-shot learning abilities without fine-tuning.

## Diffusion Models
Denoising diffusion probabilistic models (DDPMs) were popularized by Ho, Jain, and Abbeel in their 2020 paper. These models generate data by gradually denoising random noise. DALL-E 2 (April 2022) and Stable Diffusion (August 2022) applied diffusion models to text-to-image generation, achieving remarkable image quality.

## Graph Neural Networks
Scarselli et al. introduced Graph Neural Networks (GNNs) in 2009. Message passing GNNs became popular after 2017 with architectures like GraphSAGE, GCN, and GAT. GNNs are used for molecular property prediction, social network analysis, and recommendation systems.

## Notable Milestones
In 2016, AlphaGo defeated Lee Sedol 4-1 in the game of Go. DeepMind's AlphaFold solved protein folding prediction in 2020. In 2022, ChatGPT reached 100 million users faster than any previous application. By 2023, large language models with over 100 billion parameters became common.

The number of parameters in state-of-the-art models has doubled approximately every 3.4 months since 2018, far outpacing Moore's Law. Total training compute has grown by a factor of 300,000 between 2012 and 2022.
