#!/usr/bin/env python3

import pandas as pd

in_repos = 'data/repositories/'
out = 'out/'

def starfork(lang):
    data = pd.read_csv(in_repos + lang + '.csv')
    with open(out + lang + '_star_fork', 'w') as text_file:
        text_file.write(
            data[['stargazers_count', 'forks_count']].to_string(index=False))

starfork('gremlin')
starfork('cypher')
starfork('sparql')
starfork('graphql')



def names(lang):
    data = pd.read_csv(in_repos + lang + '.csv')
    with open(out + 'dataset-full_name-' + lang + '-total.csv', 'w') as text_file:
        text_file.write(
            data['repo'].to_string(index=False))

names('gremlin')
names('cypher')
names('sparql')
names('graphql')
