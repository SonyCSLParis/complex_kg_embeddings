# -*- coding: utf-8 -*-
"""
Various tests

To run (from folder root):
```bash
      python -m unittest -v tests.py
```
"""
import os
import unittest
import pandas as pd
from rdflib import Graph
from representations import KGRepresentationsConverter, read_nt
from prep_data import get_files

FOLDER = "kg_test"
FILE_P = os.path.join(FOLDER, "cg_output.nt")
GRAPH = Graph()
GRAPH.parse(FILE_P)
DF = read_nt(file_p=FILE_P)
CONVERTER = KGRepresentationsConverter()

def same_data_in_df(df1, df2):
    """ Comparing whether the data in the two df is the same (regardless of row order) """
    return df1.sort_values(
        by=df1.columns.tolist()).reset_index(drop=True) \
            .equals(
                df2.sort_values(
                    by=df2.columns.tolist()).reset_index(drop=True))

class TestKGRepr(unittest.TestCase):
    """
    Test class for KG representations converter
    """
    def test_simple_rdf_prop(self):
        """ Simple RDF with properties """
        output_c = CONVERTER.to_simple_rdf_prop(graph=GRAPH)
        output_true = read_nt(os.path.join(FOLDER, "simple_rdf_prop.nt"))

        self.assertTrue(
            same_data_in_df(df1=output_c, df2=output_true)
        )

    def test_simple_rdf_sp(self):
        """ Simple RDF with Singleton Property """
        output_c, cache = CONVERTER.to_simple_rdf_sp(graph=GRAPH, cache={})
        output_true = read_nt(os.path.join(FOLDER, "simple_rdf_sp.nt"))
        self.assertTrue(
            same_data_in_df(df1=output_c, df2=output_true)
        )
        # len(cache): # of new properties created
        self.assertEqual(len(cache), 2)

    def test_simple_rdf_reification(self):
        """ Simple RDF with Reification """
        output_c, nb = CONVERTER.to_simple_rdf_reification(graph=GRAPH, statement_nb=1)
        output_c.to_csv("test.csv", header=None, index=False, sep=" ")
        output_true = read_nt(os.path.join(FOLDER, "simple_rdf_reification.nt"))
        self.assertTrue(
            same_data_in_df(df1=output_c, df2=output_true)
        )
        # nb-1 represents the number of statements in the graph
        self.assertEqual(nb-1, 3)

    def test_hypergraph_bn(self):
        """ Hypergraph with Blank Nodes """
        output_c = CONVERTER.to_hypergraph_bn(graph=GRAPH)
        file_p = os.path.join(FOLDER, "hypergraph_bn.txt")
        output_true = pd.read_csv(file_p, sep=' ', header=None)
        self.assertTrue(
            same_data_in_df(df1=output_c, df2=output_true)
        )

    def test_hyper_relational_rdf_star(self):
        """ Hyper-relational with RDF-Star """
        output_c = CONVERTER.to_hyper_relational_rdf_star(graph=GRAPH)
        file_p = os.path.join(FOLDER, "hyper_relational_rdf_star.txt")
        output_true = pd.read_csv(file_p, sep=' ', header=None)
        self.assertTrue(
            same_data_in_df(df1=output_c, df2=output_true)
        )


class TestPrepData(unittest.TestCase):
    def test_get_files(self):
        """ Checking that the 'get_files' function retrieves the expected files """
        options = [
            {"prop": 0, "subevent": 0, "text": 0},
            {"prop": 1, "subevent": 0, "text": 0},
            {"prop": 0, "subevent": 1, "text": 0},
            {"prop": 0, "subevent": 0, "text": 1},
            {"prop": 1, "subevent": 1, "text": 0},
            {"prop": 1, "subevent": 0, "text": 1},
            {"prop": 0, "subevent": 1, "text": 1},
            {"prop": 1, "subevent": 1, "text": 1},
        ]
        files = [
            ['kg_base.nt'],
            ['kg_base.nt', 'kg_base_prop.nt'],
            ['kg_base.nt', 'kg_base_subevent.nt'],
            ['kg_base.nt', 'kg_base_text.nt'],
            ['kg_base.nt', 'kg_base_subevent_prop.nt', 'kg_base_prop.nt', 'kg_base_subevent.nt'],
            ['kg_base.nt', 'kg_base_prop.nt', 'kg_base_prop_text.nt', 'kg_base_text.nt'],
            ['kg_base.nt', 'kg_base_subevent_text.nt', 'kg_base_subevent.nt', 'kg_base_text.nt'],
            ['kg_base.nt', 'kg_base_subevent_text.nt', 'kg_base_subevent_prop.nt', 'kg_base_prop.nt', 'kg_base_prop_text.nt', 'kg_base_subevent_prop_text.nt', 'kg_base_subevent.nt', 'kg_base_text.nt']
        ]
        for i, option in enumerate(options):
            func_files = get_files(options=option, role=0, role_syntax=None)
            self.assertEqual(set(func_files), set(files[i]))
        
        option = {"prop": 0, "subevent": 0, "text": 0}
        syntax_to_files = {
            x: [f"kg_base_role_{x}.nt", "kg_base.nt"] for x in \
            ["simple_rdf_prop",
             "simple_rdf_sp", "simple_rdf_reification"]
        }
        syntax_to_files.update({
            x: [f"kg_base_role_{x}.txt", "kg_base.nt"] for x in \
            ["hypergraph_bn", "hyper_relational_rdf_star"]
        })
        for x, true_files in syntax_to_files.items():
            func_files = get_files(options=option, role=1, role_syntax=x)
            self.assertEqual(set(func_files), set(true_files))
