# Data Set Archive and Documentation
#### [**Empirical Study on the Usage of Graph Query Languages in Open Source Java Projects**](http://softlang.uni-koblenz.de/graphqls/)

This archive contains the data sets resulting from the aformentioned work, as well as scripts to generate all visualizations
that occur in the referred paper. More information on methodology and results can be found in the related publication.

#### Structure of this repository
```bash
├── data
│   ├── concrete-query-counts.csv
│   ├── dependencies
│   │   └── cypher, graphql, gremlin, mysql, sparql, xquery
│   ├── history
│   │   └── stoys_commits.json, stoys_repositories.json
│   ├── manual
│   │   └── manual_labels.csv
│   └── repositories
│       └── cypher.csv, graphql.csv, gremlin.csv, mysql.csv, sparql.csv, xquery.csv
├── lib
│   └── venn.py
├── Makefile
├── README.md
└── src
    └── activity.py, barplot_kinds.py, dataset.py, grid.py, star.sc, vdiag.py
```

#### data/dependencies

The collection of initial dependencies used in our study and obtained via querying [mvnrepository](https://mvnrepository.com/).
Files for each language list one dependency per row, where each row is structued as `<groupid>+<artifactid>`.

#### data/repositories

Including the files `cypher.csv`, `sparql.csv`, `gremlin.csv`, `graphql.csv` as well as `mysql.csv` and `xquery.csv`,
these data sets contain all repositories and metadata we obtained for the respective query language.
The CSV files have the following structure:

- `repo`: The repository ID. We ommit the common prefix https://github.com
- `origin`: Dependencies for which the repository was found (see also data/dependencies). Multiple matches are joined via `&`
- `default_branch`: The default branch of the repository
- `stargazers_count`: Number of stargazers ('stars')
- `forks_count`: Number of forks
- `updated_days_ago`, `pushed_days_ago`, `created_days_ago`: Last update (to repository), most recent push and creation date
in days before the reference date: 13.06.2019

#### data/history

All commits we processed as a JSON file `stoys_commits.json`. Running the `src/activity.py` script
generates a CSV dump with the following structure.

- `address`: The repository ID. Can be extended to https://github.com/$repo/ for access to the repository.
- `date`: Date of the commit.
- `delta`: Number of (relevant) changes.
- `email`: User that submitted the commit.
- `name`: Username of that user.
- `scope`: Scope of the change.
- `sha`: Sha of the commit.
- `type`: Type of commit: Related to Cypher or SPARQL; Change to Java or dedicated query file.

Secondly, `stoys_repositories.json` (which again can be CSV dumped via the `src/activity.py` script,
containing the following data. Some fields are shared with the previous files
and are omitted in this description.

- ...
- `commits`: Number of commits.
- `count_developers`: Number of developers.
- `count_files`: Number of files.
- `count_methods`: Number of methods.
- `exception`: InformationFor about exceptions, if a any.
- `ql`: Cypher or SPARQL

#### data/manual

Manually labelled repositories (related to the second research question in the referred paper). This
contains the top 75 highest ranking SPARQL related repositories, which were classified as either
irrelevant (42), concrete (12) or meta (21).

- `label`: With values 0 (irrelevant), 1 (concrete) and 2 (meta).
- `repo`: The repository URL.
- `branch`: The observed branch of the repository.
- `comment`: Description of the repository and SPARQL usage, recorded by reviewers.

#### data/concrete-query-counts.csv

This data set contains the counts of SELECT, ASK, CONSTRUCT and DESCRIBE SPARQL queries in
all 12 'concrete' ranked repositories.

#### Building derived data sets and visualizations
The repository includes various scripts to extract data from the source datasets in the
`data` folder. One script requires [Ammonite](http://ammonite.io/) (and Scala), all other require Python.
The supplied Makefile can (and should) be used to build


