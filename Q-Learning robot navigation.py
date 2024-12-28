# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:06:43 2024

@author: TechnoLEDs
"""

import pygame
import numpy as np
import random

# Configurações do Pygame
pygame.init()
width, height = 500, 400  # Tamanho da janela
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Navegação do Robô')

# Definir o tamanho do grid
grid_rows = 3  # 3 linhas
grid_cols = 4  # 4 colunas
cell_size = width // grid_cols  # Tamanho das células

# Definir o grid do ambiente
grid = np.array([
    [0, 0, 0, 1],  # Objetivo no canto superior direito
    [0, -1, 0, 0],  # Obstáculo no centro
    [0, 0, -1, 0]   # Obstáculo no canto inferior direito
])

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Função para desenhar o grid
def draw_grid():
    for x in range(grid_rows):
        for y in range(grid_cols):
            color = WHITE
            if grid[x, y] == -1:
                color = BLACK  # Obstáculo
            elif grid[x, y] == 1:
                color = GREEN  # Objetivo
            pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, (0, 0, 0), (y * cell_size, x * cell_size, cell_size, cell_size), 1)

# Função para desenhar o robô
def draw_robot(x, y):
    pygame.draw.circle(screen, BLUE, (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2), cell_size // 3)

# Função para obter o próximo estado com base em uma ação
def get_next_state(state, action):
    x, y = state
    if action == "up":
        x -= 1
    elif action == "down":
        x += 1
    elif action == "left":
        y -= 1
    elif action == "right":
        y += 1
    # Verificar se o próximo estado está dentro dos limites e não há obstáculos
    if 0 <= x < grid_rows and 0 <= y < grid_cols and grid[x, y] != -1:
        return (x, y)
    return state  # Se não for válido, retorna o estado atual

# Função para escolher uma ação com base na política epsilon-greedy
def choose_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploração: escolher uma ação aleatória
    else:
        # Exploração: escolher a ação com maior valor Q
        state_index = (state[0], state[1])  # Usar o estado para buscar a Q-Table
        return actions[np.argmax(q_table[state_index])]  # Melhor ação

# Função para treinar o agente usando Q-Learning
def train_q_learning():
    q_table = np.zeros((grid_rows, grid_cols, len(actions)))  # Inicializar a Q-Table com zeros
    for episode in range(num_episodios):
        state = (2, 0)  # Estado inicial (linha 2, coluna 0)
        total_reward = 0
        while state != (0, 3):  # Objetivo no canto superior direito
            action = choose_action(state, q_table, epsilon)
            next_state = get_next_state(state, action)

            # Calcular a recompensa
            if next_state == (0, 3):  # Se o objetivo foi alcançado
                reward = 1
            elif grid[next_state[0], next_state[1]] == -1:  # Se bateu em um obstáculo
                reward = -1
            else:
                reward = 0  # Recompensa neutra para movimento válido

            total_reward += reward

            # Atualizar a Q-Table usando a equação do Q-learning
            state_index = (state[0], state[1])
            next_state_index = (next_state[0], next_state[1])
            q_table[state_index][actions.index(action)] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][actions.index(action)])

            state = next_state  # Atualizar o estado para o próximo estado

    return q_table

# Definir parâmetros do Q-Learning
alpha = 0.1  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
epsilon = 0.3  # Probabilidade de exploração
num_episodios = 1000  # Número de episódios de treinamento
actions = ['up', 'down', 'left', 'right']  # Ações possíveis

# Treinamento do agente
q_table = train_q_learning()
print("Q-Table aprendida:")
print(q_table)

# Função para simular a navegação após o treinamento
def simulate_navigation(start_state, goal_state, q_table):
    state = start_state
    path = [state]
    while state != goal_state:
        action = choose_action(state, q_table, 0)  # Não explorar, usar apenas a política aprendida
        state = get_next_state(state, action)
        path.append(state)
    return path

# Testar o agente treinado
path = simulate_navigation((2, 0), (0, 3), q_table)
print("Caminho percorrido:", path)

# Animação do movimento do robô
def animate_robot(path):
    running = True
    for state in path:
        for event in pygame.event.get():  # Verificar eventos
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:  # Se pressionar uma tecla, interromper
                if event.key == pygame.K_ESCAPE:  # Pressionar ESC para parar
                    running = False
                    break
        if not running:
            break
        screen.fill(WHITE)  # Limpar a tela
        draw_grid()  # Desenhar o grid
        draw_robot(state[0], state[1])  # Desenhar o robô na nova posição
        pygame.display.flip()  # Atualizar a tela
        pygame.time.wait(500)  # Esperar 500ms para o próximo movimento

    return running

# Chamar a animação
running = animate_robot(path)

# Manter a janela aberta após a animação
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
