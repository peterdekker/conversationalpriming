# conversational-priming


## Evaluation script
In normal mode, give the variable parameter and its parameter values you want to evaluate. All parameters not given are instantiated with fixed defaults. The script will output seperate files for communicated stats and internal stats (on the y-axis), for 1sg/3sg, for the different values of the variable parameter (x-axis). The graph will show lines for innovative, conservative speakers and the total. E.g.:

```python3 evaluation.py --repeats False True```

In 'contrast persons' mode, the script will output one graph, where 1sg and 3sg are contrasted. Only lines for innovative and conservative speakers are shown, no total. No variable parameters can be given. Every model parameter you give (with only one value per parameter) will be interpreted as a fixed parameter. E.g. to create a 1sg-3sg contrast plot for a friend network:
```python3 evaluation.py --contrast_persons --friend_network True```


## Data analysis
With the script ``analyze_data.py``, an analysis of the rate of change of person markers for different grammatical persons is performed.

The script requires a few installation steps. The script uses R, so install R and CMake beforehand. On an Ubuntu machine:
```
sudo apt install r-base cmake
```
Install the Python package requirements:
```pip3 install -r requirements.txt``` (or with ``sudo`` for global install)

Install the required R packages:
```python3 install_r_pkgs.py``` (or with ``sudo`` for global install)

Now you can run the script with ```python analyze_data.py```. It will output plots in the ``output_data`` directory. This script requires the file ``verbal_person-number_indexes_merged.csv`` from https://zenodo.org/record/6028260 to be in the folder ``data``. 