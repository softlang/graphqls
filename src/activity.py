#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import statsmodels.api as sm
import math
import statsmodels.stats.anova as anova
from functools import reduce
import datetime as dt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

pd.options.display.max_rows = 30
pd.options.display.max_columns = 45
pd.options.display.width = 900


def write_table(table, target, crop1, crop2):
    output = table.to_latex(index=False).splitlines()
    f = open(target, "w+")
    for s in output[crop1:len(output) - crop2]:
        f.write(s + "\n")
    f.close()


def writeCommands(dict, target):
    f = open(target, "w+")
    for key, value in dict.items():
        f.write('\def \\gencmd' + key.replace(' ', '').replace('_', '') + '{' + str(value) + '}\n')
    f.close()


sns.set_context("paper")
sns.set(font_scale=1.0)
sns.set_style("whitegrid")

commands = {}

pallet = sns.cubehelix_palette(5, start=2.8, rot=.1)

in_history = 'data/history/'
in_repos = 'data/repositories/'

out_history = 'out/'
target_figure = ''
target = 'out/'

query_colors = {'sparql': '#0b459d',
                'sparql-java': '#0b459d',
                'sparql-file': '#0b459d',
                'cypher': '#008abc',
                'mysql': '#FFa500',
                'cypher-file': '#008abc',
                'cypher-java': '#008abc',
                'graphql': '#e535ab',
                'gremlin': '#006d00',
                'xquery': '#FF0000'}

query_markers = {'sparql-java': '-',
                 'sparql-file': ':',

                 'cypher-java': '-',
                 'cypher-file': ':'}

roller = '120D'

total_repositories = 16801

positons = {'address': 2, 'email': 3, 'scope': 4}
star_threshold = 1

fig, axs = plt.subplots(5, 2, sharex='row', sharey='row', figsize=(14, 14))

repositories = pd.read_json(in_history + 'stoys_repositories.json', lines=True).rename(columns={'repo': 'address'})
repositories = repositories[
    repositories['exception'].apply(lambda x: not str(x).startswith('org.eclipse.jgit.api.errors.TransportException'))]
repositories.to_csv(out_history + 'history_repositories.csv', index=False)

commits = pd.read_json(in_history + 'stoys_commits.json', lines=True)
commits.to_csv(out_history + 'history_commits.csv', index=False)
commits = pd.merge(repositories, commits)


# Exceptions need to be depicted in the paper.
exception = repositories['exception'].value_counts()
commands['exceptions'] = str(exception.sum())
print(exception)

# Sample fraction.
sample_fraction = float(len(repositories)) / float(total_repositories)
commands['samplefraction'] = str(int(sample_fraction * 100))

# Outlier extraction on commits and repositories by number of scopes touched.
scope_sha_threshold = 32
sha_scopes = commits.groupby('sha')['scope'].nunique().to_frame('n_sha_scopes').reset_index()
commands['numbercommitswithquery'] = str(int(len(sha_scopes) / 1000)) + '.000'
commands['numbercommitswithqueryexcuded'] = str(
    round((float(len(sha_scopes[sha_scopes['n_sha_scopes'] >= scope_sha_threshold])) / float(len(sha_scopes))) * 100,
          2))

commits = pd.merge(commits, sha_scopes)
commits = pd.merge(commits, commits.groupby('address')['scope'].nunique().to_frame('n_repository_scopes').reset_index())

# Filtering of extremes

commits = commits[(commits['n_sha_scopes'] < scope_sha_threshold)]

lowlim = commits['date'].min()
highlim = commits['date'].max()


def plot_with_threshold(hax, data):
    data = data.pivot_table(index=['date', 'address', 'scope', 'email'], columns='type', values='delta', aggfunc='sum')
    data = data.fillna(0.0).applymap(lambda x: x != 0.0)

    new_index = pd.MultiIndex.from_tuples([(x, y) for x in data.index.names[1:4] for y in data.columns],
                                          names=['m', 'type'])

    data = data.apply(lambda row: pd.Series([x if y else np.NaN for x in row.name[1:4] for y in row], index=new_index),
                      axis=1)
    data = data.apply(lambda x: pd.factorize(x)[0], axis=0)
    data = data.reset_index().set_index('date').sort_index()
    data = data.drop([('address', ''), ('scope', ''), ('email', '')], axis=1)
    data = data.rolling(roller, min_periods=0).apply(lambda x: len(np.unique(x[x != -1]))).astype(int)
    data = data.stack(level=0).stack(level=0).reset_index()

    for (m, tpe), data in data.groupby(['m', 'type']):
        data = data.set_index('date').drop(['m', 'type'], axis=1)
        ax = axs[positons[m]][hax]
        data.plot(kind='line', ax=ax, color=query_colors[tpe], style=query_markers[tpe], label=tpe, lw=1.4, alpha=.99)
        ax.set_ylabel(m)
        ax.set_xlabel('')
        ax.set_xlim([lowlim, highlim])
        ax.legend().set_visible(False)


plot_with_threshold(0, commits)
plot_with_threshold(1, commits[commits['stargazers_count'] > star_threshold])

gremlin = pd.read_csv(in_repos + 'gremlin.csv')
sparql = pd.read_csv (in_repos + 'sparql.csv')
cypher = pd.read_csv (in_repos + 'cypher.csv')
xquery = pd.read_csv (in_repos + 'xquery.csv')
graphql = pd.read_csv(in_repos + 'graphql.csv')
mysql = pd.read_csv  (in_repos + 'mysql.csv')

gremlin['type'] = 'gremlin'
sparql['type'] = 'sparql'
cypher['type'] = 'cypher'
xquery['type'] = 'xquery'
graphql['type'] = 'graphql'
mysql['type'] = 'mysql'

languages = pd.concat([gremlin, sparql, cypher, xquery, graphql, mysql])

languages['created'] = pd.to_datetime('2019-06-13T00:00:00.000Z') - languages['created_days_ago'].apply(
    lambda x: pd.Timedelta(str(x) + ' days'))

for tpe, l in languages.groupby('type'):
    l.groupby('created')['repo'].count().cumsum().plot(kind='line', ax=axs[1][0], color=query_colors[tpe],
                                                       label=tpe)

for tpe, l in languages[languages['stargazers_count'] > star_threshold].groupby('type'):
    l.groupby('created')['repo'].count().cumsum().plot(kind='line', ax=axs[1][1], color=query_colors[tpe])

languages_low = languages[(languages['type'] != 'mysql')]

for tpe, l in languages_low.groupby('type'):
    l.groupby('created')['repo'].count().cumsum().plot(kind='line', ax=axs[0][0], color=query_colors[tpe],
                                                       label=tpe)

for tpe, l in languages_low[languages_low['stargazers_count'] > star_threshold].groupby('type'):
    l.groupby('created')['repo'].count().cumsum().plot(kind='line', ax=axs[0][1], color=query_colors[tpe])

fraction_stars_filtered = float(
    (len(languages[languages['stargazers_count'] > star_threshold])) / float(len(languages)))

commands['fractionstarsfiltered'] = str(int(fraction_stars_filtered * 100))

axs[0][0].set_xlim([lowlim, highlim])
axs[0][1].set_xlim([lowlim, highlim])
axs[1][0].set_xlim([lowlim, highlim])
axs[1][1].set_xlim([lowlim, highlim])

axs[1][0].set_ylabel('total repositories [log]')
axs[0][0].set_ylabel('total repositories')
axs[positons['address']][0].set_ylabel('active repositories\n(sample ' + str(int(sample_fraction * 100)) + '%)')
axs[positons['email']][0].set_ylabel('active developers\n(sample ' + str(int(sample_fraction * 100)) + '%)')
axs[positons['scope']][0].set_ylabel('active artifacts\n(sample ' + str(int(sample_fraction * 100)) + '%)')

axs[0][0].set_title(str(len(languages)) + ' GitHub repositories with queries')
axs[0][1].set_title(
    '... and more than ' + str(star_threshold) + ' stars (' + (str(int(fraction_stars_filtered * 100)) + '%)'))

axs[1][0].set_yscale('log')
axs[1][1].set_yscale('log')

patchesql = [mpatches.Patch(color=query_colors['mysql'], label='MySQL'),
             mpatches.Patch(color=query_colors['cypher'], label='Cypher'),
             mpatches.Patch(color=query_colors['sparql'], label='SPARQL'),
             mpatches.Patch(color=query_colors['xquery'], label='XQuery'),
             mpatches.Patch(color=query_colors['graphql'], label='GraphQL'),
             mpatches.Patch(color=query_colors['gremlin'], label='Gremlin'),
             ]

axs[0][0].legend(handles=patchesql)

patchesgr = [mpatches.Patch(color=query_colors['sparql'], label='SPARQL (window 120 days)'),
             mpatches.Patch(color=query_colors['cypher'], label='Cypher (window 120 days)'),
             mlines.Line2D([], [], color='k', linestyle=':', label='in query files'),
             mlines.Line2D([], [], color='k', linestyle='-', label='in Java methods')]

axs[2][0].legend(handles=patchesgr)


# Add  annotations.


def annotate(text, date):
    alpha = 0.5
    axs[4][0].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[3][0].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[2][0].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[1][0].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[0][0].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[4][1].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[3][1].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[2][1].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[1][1].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)
    axs[0][1].axvline(x=date, linestyle='dashed', c='k', alpha=alpha)

    axs[0][0].text(x=date, y=8200, s=text, alpha=0.9, color='k')
    axs[0][1].text(x=date, y=8200, s=text, alpha=0.9, color='k')


# Oct 2007 : Github release
annotate("   Github release\n", dt.datetime(2007, 10, 1))
# Jun 2011 : (ca.) Introduction Cypher to Neo4j
annotate(" Cypher\n Neo4j", dt.datetime(2011, 6, 1))
# Oct 2015 : openCypher founded
annotate(" openCypher", dt.datetime(2015, 10, 1))
# Jan 2008 : W3C Recommentation of SPARQL 1.0
annotate("   SPARQL 1.0", dt.datetime(2008, 1, 1))
# Mar 2013 : W3C REcommentation of SPARQL 1.1
annotate(" SPARQL 1.1", dt.datetime(2013, 3, 1))

# ymin, ymax = axs[3][0].get_ylim()
# axs[3][0].set_ylim([offset, ymax])
# axs[3][1].set_ylim([offset, ymax])

fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.07, hspace=0.06, wspace=0.03)
fig.savefig(target_figure + 'evolution.pdf', bbox_inches='tight')

# ---------- Linear regression ------------------------------------------------------------------------------------

repositories = pd.read_json(in_history + 'stoys_repositories.json', lines=True).rename(columns={'repo': 'address'})
repositories = repositories[
    repositories['exception'].apply(lambda x: not str(x).startswith('org.eclipse.jgit.api.errors.TransportException'))]

# Add number without filtering.
commands['numbersamplerepos'] = str(len(repositories))
commands['fractionerror'] = str(round(float(exception.sum()) / float(len(repositories)) * 100, 1))

repositories = repositories[repositories.exception.isnull()]
commits = pd.read_json(in_history + 'stoys_commits.json', lines=True)

repositories['files'] = repositories['count_files']
repositories['methods'] = repositories['count_methods']

repositories['is_sparql'] = (repositories['ql'] == 'sparql').astype(int)
repositories = repositories[repositories['exception'].isnull()]

repositories = pd.merge(repositories, commits.groupby('address')['sha'].nunique().to_frame('n_sha_query').reset_index(),
                        how='left')
repositories = pd.merge(repositories,
                        commits.groupby('address')['email'].nunique().to_frame('active_developers').reset_index(),
                        how='left')
repositories = pd.merge(repositories,
                        commits[commits['type'].apply(lambda x: x.endswith('file'))].groupby('address')[
                            'scope'].nunique().to_frame('active_files').reset_index(), how='left')

repositories = pd.merge(repositories,
                        commits[commits['type'].apply(lambda x: x.endswith('java'))].groupby('address')[
                            'scope'].nunique().to_frame('active_methods').reset_index(), how='left')

repositories['query_activity'] = repositories['n_sha_query'] / repositories['commits']
repositories['query_activity'] = repositories['query_activity'].fillna(0.0)
repositories['active_developers'] = repositories['active_developers'].fillna(0.0)
repositories['active_files'] = repositories['active_files'].fillna(0.0)
repositories['active_methods'] = repositories['active_methods'].fillna(0.0)
repositories['query_commits'] = repositories['n_sha_query'].fillna(0.0)

print(repositories)

models = [
    ['scale(np.log(created_days_ago))', 'scale(np.log(stargazers_count+0.5))'],
    ['scale(np.log(active_developers +0.5))'],
    ['scale(np.log(active_files +0.5))'],
    # ['scale(np.log(active_methods +0.5))'],
    # ['scale(np.log(query_commits+0.5))'],
]

params = []
pvalues = []
prsquared = []
factors = []
vifmax = 0.0
corrmax = 0.0

mdls = []

for add in models:
    factors = factors + add
    y, X = patsy.dmatrices('is_sparql~ ' + reduce(lambda x, y: x + ' + ' + y, factors), data=repositories,
                           return_type='dataframe')

    mod = sm.OLS(y, X)
    res = mod.fit()
    mdls.append(res)
    print(res.summary())
    params.append(res.params)
    pvalues.append(res.pvalues)
    prsquared.append(res.rsquared)


    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    corr = X.corr()
    print(vif)
    print(corr)

    vifmax = max(vifmax, (vif['VIF Factor'].max()))
    corrmax = max(corrmax, corr.applymap(lambda x: x if x != 1.0 else 0.0).max().max())

print(anova.anova_lm(*mdls, typ=1))

params = pd.concat(params, axis=1)
pvalues = pd.concat(pvalues, axis=1)
params = params.applymap(lambda x: str(round(x, 3)) if not pd.isnull(x) else "")
pvalues = pvalues.applymap(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else '')))

combined = params.combine(pvalues, lambda x, y: x + y)
combined.loc['R-squ.'] = [round(x, 3) for x in prsquared]
combined.columns = ['Model ' + str(x) for x in combined.columns]

commands['vifmax'] = str(math.ceil(vifmax))
commands['corrmax'] = str(math.ceil(corrmax * 10) / 10)

# Other measures.
commands['totalcommits'] = str(int(repositories['commits'].sum() / 1000)) + '.000'

write_table(combined.drop('Intercept', axis=0).reset_index(), target + '/lr.tex', 4, 2)
writeCommands(commands, target + '/gencmd.tex')
