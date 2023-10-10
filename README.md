# Conversational priming in repetitional responses:
This repository contains the code for the agent-based model (in folder ''agents'') and the data analysis scripts for the paper ``Mechanisms of change in person markers: Conversational priming versus frequency''.

## Agent-based model
In normal mode, give the variable parameter and its parameter values you want to evaluate. All parameters not given are instantiated with fixed defaults. The script will output seperate files for communicated stats and internal stats (on the y-axis), for 1sg/3sg, for the different values of the variable parameter (x-axis). The graph will show lines for innovative, conservative speakers and the total. E.g.:

```python3 evaluation.py --repeats False True```

In 'contrast persons' mode, the script will output one graph, where 1sg and 3sg are contrasted. Only lines for innovative and conservative speakers are shown, no total. No variable parameters can be given. Every model parameter you give (with only one value per parameter) will be interpreted as a fixed parameter. E.g. to create a 1sg-3sg contrast plot for a friend network:
```python3 evaluation.py --contrast_persons --friend_network True```


