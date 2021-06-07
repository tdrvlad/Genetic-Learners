import os, json
import matplotlib.pyplot as plt
import numpy as np

logs_dir = 'logs_run_1'

best_individual_file = os.path.join(logs_dir, 'Best_Individual.json')
evolution_overview_file = os.path.join(logs_dir, 'Evolution_Overview.json')

with open(evolution_overview_file) as json_file:
    data = json.load(json_file)

average_loss_generation = []
for key, value in data.items():
    val = float(value.split(' ')[1])
    average_loss_generation.append(val)
average_loss_generation = np.delete(average_loss_generation, np.argmax(average_loss_generation))


print(average_loss_generation)
plt.figure()
plt.plot(average_loss_generation)
plt.savefig(os.path.join(logs_dir, 'EvolutionOverview.png'))
