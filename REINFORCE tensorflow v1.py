# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:11:25 2025

@author: TechnoLEDs

Arrumar o código!!
"""

import tensorflow as tf
# import keras
import gymnasium as gym
import numpy as np

# Definindo a rede neural para a política (actor)
class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# REINFORCE Algorithm
def reinforce(env, policy_network, optimizer, episodes=1000, gamma=0.99):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            state = tf.convert_to_tensor(state, dtype=tf.float32)  # Convertendo para tensor
            action_probs = policy_network(state[None, :])  # Adicionando uma dimensão extra
            action = np.random.choice(len(action_probs[0]), p=action_probs[0].numpy())  # Selecionando ação
            log_prob = tf.math.log(action_probs[0, action])  # Log da probabilidade da ação
            log_probs.append(log_prob)

            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)

        # Calcular o retorno descontado (G_t)
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        # Normalizando os retornos para estabilidade
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-5)

        # Atualizar a política usando o gradiente
        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(tf.convert_to_tensor(log_probs) * tf.convert_to_tensor(returns))  # REINFORCE Update Rule

        grads = tape.gradient(loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Loss: {loss.numpy()}")

# Configuração do ambiente e do modelo
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

policy_network = PolicyNetwork(input_size, output_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

# Treinando o agente com REINFORCE
reinforce(env, policy_network, optimizer, episodes=1000)
