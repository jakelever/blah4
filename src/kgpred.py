#!/usr/bin/env python
import argparse
import collections
import csv
import gzip
import itertools
import logging
import os

import numpy as np
from scipy.sparse import dok_matrix, save_npz

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    TODO
    ''')
    parser.add_argument('entity_counts_file',
                        help='TODO')
    parser.add_argument('relation_counts_file',
                        help='TODO')
    parser.add_argument('output_pair',
                        help='TODO')
    parser.add_argument('--cutoff_year', type=int, default=2010,
                        help='TODO')
    parser.add_argument('--weighting_exponent', type=float, default=0.6,
                        help='TODO')
    args = parser.parse_args()
    return args.entity_counts_file, args.relation_counts_file, args.output_pair, \
           args.cutoff_year, args.weighting_exponent


def main():
    logging.basicConfig(level=logging.INFO)
    entity_counts_file, relation_counts_file, output_dir, cutoff_year, weighting_exponent = parse_parameters()

    types = ['Chemical', 'Disease', 'Gene']
    type_to_name_to_count = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(entity_counts_file, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        for row in reader:
            year, type_1, name_1, count = row
            if int(year) >= cutoff_year:
                continue
            if type_1 not in types:
                raise ValueError(f'Unknown entity type {type_1}.')
            type_to_name_to_count[type_1][name_1] += int(count)
    for type_1, names_to_count in type_to_name_to_count.items():
        logging.info(f'Found {len(names_to_count)} entities of type {type_1}.')

    relations = list(itertools.combinations(types, 2))
    relations_to_names_to_count = collections.defaultdict(lambda: collections.defaultdict(int))
    with gzip.open(relation_counts_file, 'rt', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        for row in reader:
            year, type_1, name_1, type_2, name_2, count = row
            if int(year) >= cutoff_year:
                continue
            if type_1 == type_2:
                continue
            if type_2 < type_1:
                relation = (type_2, type_1)
                name_1, name_2 = name_2, name_1
            else:
                relation = (type_1, type_2)
            pair = (name_1, name_2)
            if relation not in relations:
                raise ValueError(f'Unknown relation {relation}.')
            relations_to_names_to_count[relation][pair] += int(count)
    for relation, names_to_count in relations_to_names_to_count.items():
        logging.info(f'Found {len(names_to_count)} entities of type {relation}.')

    for relation, names_to_count in relations_to_names_to_count.items():
        # compute normalization factor for each relation type, i.e., the total
        # number of co-mentions across all entity pairs
        total_counts = sum((count for count in names_to_count.values()))

        file_name = f'{relation[0]}_{relation[1]}_cutoffyear_{cutoff_year}_weightingexponent_{weighting_exponent}'
        output_path = os.path.join(output_dir, file_name + '.tsv.gz')

        row_labels_path = os.path.join(output_dir, file_name + '_rows.txt.gz')
        column_labels_path = os.path.join(output_dir, file_name + '_columns.txt.gz')
        row_labels = sorted(set([names[0] for names in names_to_count.keys()]))
        column_labels = sorted(set([names[1] for names in names_to_count.keys()]))
        row_label_to_index = {row_label: i for i, row_label in enumerate(row_labels)}
        column_label_to_index = {column_label: i for i, column_label in enumerate(column_labels)}
        with gzip.open(row_labels_path, 'wt') as row_out:
            for row_label in row_labels:
                row_out.write(row_label + os.linesep)
        with gzip.open(column_labels_path, 'wt') as column_out:
            for column_label in column_labels:
                column_out.write(column_label + os.linesep)

        matrix_path = os.path.join(output_dir, file_name + '.npz')
        matrix = dok_matrix((len(row_labels), len(column_labels)), dtype=np.float32)
        with gzip.open(output_path, 'wt') as output_file:
            for names, count in names_to_count.items():
                for type_name in zip(relation, names):
                    type_1, name_1 = type_name
                    if name_1 not in type_to_name_to_count[type_1]:
                        raise ValueError(f'Unknown entity {name_1} of type {type_1}.')
                co_occurrence = (count ** weighting_exponent) * \
                                (((count * total_counts) /
                                    (type_to_name_to_count[relation[0]][names[0]] *
                                     type_to_name_to_count[relation[1]][names[1]])) ** (1 - weighting_exponent))
                row = [relation[0], names[0], relation[1], names[1], str(co_occurrence)]
                output_file.write('\t'.join(row) + os.linesep)
                matrix[row_label_to_index[names[0]], column_label_to_index[names[1]]] = co_occurrence
        save_npz(matrix_path, matrix.tobsr())


if __name__ == '__main__':
    main()
