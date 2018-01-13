#!/usr/bin/env nextflow

entity_counts = file("../../data/20180111-graph-train-test/entityCounts.subset.tsv")
relation_counts = file("../../data/20180111-graph-train-test/relationCounts.subset.tsv.gz")
weighting_exponents = Channel.from(0.1, 0.25, 0.5, 0.75, 0.9)
cutoff_years = Channel.from(2005, 2010)

process coscores {
    publishDir 'run', mode: 'copy'
    cpus 1
    executor 'local'
    maxForks 2

    input:
    file entity_counts
    file relation_counts
    each weighting_exponent from weighting_exponents
    each cutoff_year from cutoff_years

    output:
    file '*z' into output_files

    """
    kgpred.py $entity_counts $relation_counts . --weighting_exponent $weighting_exponent --cutoff_year $cutoff_year
    """
}
