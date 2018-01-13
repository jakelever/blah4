# Literature-based Knowledge Discovery with Relations

This repository contains code for a project started at the BLAH4 hackathon that took place in Kashiwa, Japan.
The goal of the projects is to predict edges in a knowledge graph that will appear in future publications

## Data set 

We use a knowledge graph built using distant supervision using:

- seed knowledge from WikiData
- text data from PubTator-annotated PubMed

The training set contains novel tuples from publications before 2010,
the test set contains novel tuples from publications from and including 2010.
The datasets contain negatives tuples that have been randomnly selected from all pairs of nodes that do not appear in the positive set to match the number of positives.

## Dependencies

- Python 3
- Scipy 1.0
- nextflow workflow manager
