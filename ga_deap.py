import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt


# Hiperparâmetros
NGEN = 100
NPOP = 50
CXPB = .7
MUTPB = .3


# Criar fitness (maximização)
creator.create('FitnessMax', base.Fitness, weights=(1.,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Variável inteira [0, 10]
toolbox.register('x', random.randint, 0, 10)

# Variável contínua [0, 10]
toolbox.register('y', random.uniform, 0, 10)

# Combinar os geradores de variável inteira e contínua
toolbox.register('individual', tools.initIterate, creator.Individual,
                 lambda: [toolbox.x(), toolbox.y()])

# Gerador de população
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Função objetivo
def funcao_objetivo(individual):
    x, y = individual
    
    # Restrições
    cons1 = (-x + 2*x*y <= 8)
    cons2 = (2*x + y <= 14)
    cons3 = (2*x - y <= 10)    
    
    # Valor da função
    z = x + x*y

    # Penalidade para violação de alguma restrição
    if not (cons1 and cons2 and cons3):
        z -= 1e3

    return z,

# Registra a função objetivo
toolbox.register('evaluate', funcao_objetivo)


# Operador de crossover que preserva identidade mixed integer
def cxMINLP(ind1, ind2, alpha=.5, indpb=.5):
    # Variável inteira x -> uniform crossover
    x1, x2 = tools.cxUniform([ind1[0]], [ind2[0]], indpb)
    ind1[0], ind2[0] = x1[0], x2[0] 
    # Variável contínua y -> blend crossover
    y1, y2 = tools.cxBlend([ind1[1]], [ind2[1]], alpha)
    ind1[1], ind2[1] = y1[0], y2[0]
    # Garante limites
    ind1[0] = int(min(max(ind1[0], 0), 10))
    ind2[0] = int(min(max(ind2[0], 0), 10))
    ind1[1] = min(max(ind1[1], 0), 10)
    ind2[1] = min(max(ind2[1], 0), 10)

    return ind1, ind2


# Operador de mutação que preserva identidade mixed integer
def mutMINLP(ind, mu=0, sigma=1, indpb=.2):
    # Variável inteira x -> uniform mutation
    x_mut, = tools.mutUniformInt([ind[0]], 0, 10, indpb)
    ind[0] = int(x_mut[0])
    # Variável contínuaa y -> gaussian mutation
    y_mut, = tools.mutGaussian([ind[1]], mu, sigma, indpb)
    ind[1] = min(max(y_mut[0], 0), 10)

    return ind,


# Registra os operadores genéticos
toolbox.register('mate', cxMINLP)
toolbox.register('mutate', mutMINLP)

# Registra o método de seleção
toolbox.register('select', tools.selTournament, tournsize=3)


# Define a rotina do algoritmo genético
def main():
    
    # Cria população inicial
    pop = toolbox.population(NPOP)

    # Registra o hall of fame (melhor indivíduo da geração)
    hof = tools.HallOfFame(1)

    # Registra estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register('max', np.max)
    stats.register('hof', lambda _: hof[0] if len(hof) > 0 else None)

    # Algoritmo evolutivo
    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
        stats=stats, halloffame=hof, verbose=True,
    )

    plt.plot(log.select('max'))
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

if __name__ == '__main__':
    main()
