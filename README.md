# Code for _Conversational priming in repetitional responses as a mechanism in language change: Evidence from agent-based modelling_
This repository contains the code for the agent-based model (in folder ''agents'') and the data analysis scripts for the paper "Conversational priming in repetitional responses as a mechanism in language change: Evidence from agent-based modelling'' by Peter Dekker, Sonja Gipper and Bart de Boer.

## Agent-based model
In normal mode, give the variable parameter and its parameter values you want to evaluate. All parameters not given are instantiated with fixed defaults. The script will output seperate files for communicated stats and internal stats (on the y-axis), for 1sg/3sg, for the different values of the variable parameter (x-axis). The graph will show lines for innovative, conservative speakers and the total. E.g.:

```python3 evaluation.py --repeats False True```

In 'contrast persons' mode, the script will output one graph, where 1sg and 3sg are contrasted. Only lines for innovative and conservative speakers are shown, no total. No variable parameters can be given. Every model parameter you give (with only one value per parameter) will be interpreted as a fixed parameter. E.g. to create a 1sg-3sg contrast plot for a friend network:
```python3 evaluation.py --contrast_persons --friend_network True --runlabel mylabel-friendnetwork```


## Calculation proportion persons in Corpus Gesproken Nederlands (CGN, Spoken Dutch Corpus)
``analyse_cgn.py``: Standalone script to count persons in Corpus Gesproken Nederlands (CGN, Spoken Dutch Corpus)
Download the annotations of the CGN corpus (full CGN corpus with sound files is not needed):

CGN-annotaties (Version 2.0.3) (2014) [Data set]. Available at the Dutch Language Institute: http://hdl.handle.net/10032/tm-a2-n5

The script expects the main 'Data' folder of the corpus to be placed in the directory ``data/CGNAnn``.