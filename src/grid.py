#!/usr/bin/env python3

import prettyplotlib as ppl
import matplotlib.pyplot as plt
from prettyplotlib import brewer2mpl
from matplotlib.colors import LogNorm
import numpy as np

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sys
sys.path.append('out')

from generated_grid import *



# --- COLORS ---

green_purple = brewer2mpl.get_map('PRGn', 'diverging', 11).mpl_colormap
red_purple = brewer2mpl.get_map('RdPu', 'Sequential', 9).mpl_colormap
greys = brewer2mpl.get_map('greys', 'Sequential', 9).mpl_colormap

def make_map(r, g, b):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, r/256, N)
    vals[:, 1] = np.linspace(1, g/256, N)
    vals[:, 2] = np.linspace(1, b/256, N)
    return ListedColormap(vals)

#sparql_colors  = make_map( 11,  69, 157)
#cypher_colors  = make_map(  0, 138, 188)
#gremlin_colors = make_map(  0, 109,   0)
#graphql_colors = make_map(229,  53, 171)

sparql_colors  = greys
cypher_colors  = greys
gremlin_colors = greys
graphql_colors = greys



# --- SETUP ---

fig = plt.figure()
grix = 2
griy = 2
normMax = max(sparql.max().max(), cypher.max().max(), gremlin.max().max(), graphql.max().max())
normMin = min(sparql.min().min(), cypher.min().min(), gremlin.min().min(), graphql.min().min())

# SPARQL

ax1 = fig.add_subplot(grix,griy,1, label="aaa")

ppl.pcolormesh(fig, ax1, sparql,
               xticklabels=[0,1,2,3,4,5,"+"],
               yticklabels=[0,1,2,3,4,5,"+"],
               cmap=sparql_colors)

for (i, j), z in np.ndenumerate(sparql_label):
    ax1.text(0.5 + j, 0.5 + i,
             '{:d}'.format(z),
             ha='center', va='center', color='white')

plt.xlabel("SPARQL forks (Ø" + sparql_forks_avg + ")")
plt.ylabel("SPARQL stars (Ø" + sparql_stars_avg + ")")

# Cypher

ax2 = fig.add_subplot(grix,griy,2)

ppl.pcolormesh(fig, ax2, cypher,
               xticklabels=[0,1,2,3,4,5,"+"],
               yticklabels=[0,1,2,3,4,5,"+"],
               cmap=cypher_colors)

for (i, j), z in np.ndenumerate(cypher_label):
    ax2.text(0.5 + j, 0.5 + i,
             '{:d}'.format(z),
             ha='center', va='center', color='white')

plt.xlabel("Cypher forks (Ø" + cypher_forks_avg + ")")
plt.ylabel("Cypher stars (Ø" + cypher_stars_avg + ")")

# Gremlin

ax3 = fig.add_subplot(grix,griy,3)

ppl.pcolormesh(fig, ax3, gremlin,
               xticklabels=[0,1,2,3,4,5,"+"],
               yticklabels=[0,1,2,3,4,5,"+"],
               cmap=gremlin_colors)

for (i, j), z in np.ndenumerate(gremlin_label):
    ax3.text(0.5 + j, 0.5 + i,
             '{:d}'.format(z),
             ha='center', va='center', color='white')

plt.xlabel("Gremlin forks (Ø" + gremlin_forks_avg + ")")
plt.ylabel("Gremlin stars (Ø" + gremlin_stars_avg + ")")

# GraphQL

ax4 = fig.add_subplot(grix,griy,4)

ppl.pcolormesh(fig, ax4, graphql,
               xticklabels=[0,1,2,3,4,5,"+"],
               yticklabels=[0,1,2,3,4,5,"+"],
               cmap=graphql_colors)

for (i, j), z in np.ndenumerate(graphql_label):
    ax4.text(0.5 + j, 0.5 + i,
             '{:d}'.format(z),
             ha='center', va='center', color='white')

plt.xlabel("GraphQL forks (Ø" + graphql_forks_avg + ")")
plt.ylabel("GraphQL stars (Ø" + graphql_stars_avg + ")")

plt.savefig('star_fork.pdf')
