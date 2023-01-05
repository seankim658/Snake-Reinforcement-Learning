# imports 
import matplotlib.pyplot as plt 
from IPython import display 

plt.ion() 

def plot(scores, mean_scores):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf() 
    plt.title('Snake Training')
    plt.xlabel('Game #')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin = 0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    # use block = False for interactive mode (per matplotlib documentation)
    plt.show(block = False)
    plt.pause(0.1)