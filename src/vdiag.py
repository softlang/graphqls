#!/usr/bin/env python3
# coding: utf-8

# ipython notebook requires this
# %matplotlib inline

# python console requires this
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import sys
sys.path.append('lib')

import venn

sparqldata  = set(line.strip() for line in open('out/dataset-full_name-sparql-total.csv'))
cypherdata  = set(line.strip() for line in open('out/dataset-full_name-cypher-total.csv'))
graphqldata = set(line.strip() for line in open('out/dataset-full_name-graphql-total.csv'))
gremlindata = set(line.strip() for line in open('out/dataset-full_name-gremlin-total.csv'))

data = [sparqldata, cypherdata, graphqldata, gremlindata]

labels = venn.get_labels(data, fill=['number'])
fig, ax = venn.venn4(labels, names=['SPARQL', 'Cypher', 'GraphQL', 'Gremlin'])
fig.savefig('vennrepos.pdf', bbox_inches='tight')
plt.close()
