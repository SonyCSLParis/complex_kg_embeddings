# -*- coding: utf-8 -*-
"""
Converting the output of ChronoGrapher into different formats
"""
import json
from rdflib import Graph
import pandas as pd
from utils import read_nt

def get_columns_from_query(query):
    """ column names from sparql query """
    res = query.split(" WHERE {")[0].split("?")
    res = [x.strip() for x in res if 'select' not in x.lower()]
    return res


def query_return_csv(graph, query):
    """ (SELECT) query to be returned as .csv """
    res = graph.query(query)
    res = pd.DataFrame(res)
    columns = get_columns_from_query(query=query)
    if res.shape[0] == 0:
        return pd.DataFrame(columns=columns)
    else:
        res.columns = columns
        return res


class KGRepresentationsConverter:
    """ Converting between different KG representations """
    def __init__(self):
        """ Store main properties """
        self.prefixes = {
            "sem": "http://semanticweb.cs.vu.nl/2009/11/sem/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "wsj": "https://w3id.org/framester/wsj/",
            "ex": "http://example.com/",
            "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
        }
        self.prefixes_query = "\n".join([f"PREFIX {k}: <{v}>" for k, v in self.prefixes.items()])

        self.queries = {
            "simple_rdf_with_properties": """
            SELECT DISTINCT ?event ?sub_event ?frame ?role ?role_type ?entity WHERE {
            ?event ?pred ?abstract .
            ?abstract nif:sentence ?sentence .
            ?sub_event wsj:fromDocument ?sentence ;
                    a wsj:CorpusEntry ;
                    wsj:onFrame ?frame ;
                    wsj:withmappedrole ?role .
            ?role wsj:withfnfe ?role_type .
            OPTIONAL {?role skos:related ?entity}
            VALUES ?pred {ex:abstract ex:hasOutcome}
            }
            """
        }
        self.queries = {k: self.prefixes_query + "\n" + val for k, val in self.queries.items()}
        self.triples_col = ["subject", "predicate", "object"]

    def link_two_cols(self, df, cols, pred):
        """ Link two cols in a df with a predicate
        cols[0] will be the subject and cols[1] the object """
        res = df[cols].drop_duplicates().dropna()
        res["predicate"] = pred
        res = res.rename(columns={cols[0]: "subject", cols[1]: "object"})
        return res[self.triples_col]

    def add_text(self, df, output):
        """ Extract triples that have literals as objects """
        nodes = list(set(list(output[self.triples_col[0]].unique()) + \
            list(output[self.triples_col[2]].unique())))
        df = df[df[self.triples_col[0]].isin(nodes)]
        df = df[~df.object.str.startswith("<")]
        return pd.concat([output, df])

    def init_sub_graphs(self, df, mode):
        """ Sub graphs common to all representation types """
        sub_graphs = []
        if mode == "role":
            sub_graphs.append(self.link_two_cols(
                df=df, cols=["sub_event", "event"], pred=f"{self.prefixes['sem']}subEventOf"))
        if mode == "causation":
            sub_graphs.append(self.link_two_cols(
                df=df, cols=["event", "sub_event"], pred=f"{self.prefixes['ex']}hasOutcome"))
        sub_graphs.append(self.link_two_cols(
            df=df, cols=["sub_event", "frame"], pred=f"{self.prefixes['rdf']}type"))
        sub_graphs.append(self.link_two_cols(
            df=df, cols=["role", "entity"], pred=f"{self.prefixes['skos']}related"))
        return sub_graphs

    def get_output_from_subgraph(self, sg):
        """ Concat sub graphs into format compatible with .nt """
        output = pd.concat(sg)
        output["."] = "."
        for col in self.triples_col:
            output[col] = output[col].apply(lambda x: f"<{x}>")
        return output

    def to_simple_rdf_prop(self, graph, mode: str = "role"):
        """ Output ChronoGrapher -> Simple RDF (roles with properties) """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])

        sub_graphs = self.init_sub_graphs(df=res, mode=mode)
        sub_graphs.append(
            res[["sub_event", "role_type", "role"]].drop_duplicates().rename(
                columns={"sub_event": "subject", "role_type": "predicate", "role": "object"}))

        return self.get_output_from_subgraph(sg=sub_graphs)

    def to_simple_rdf_sp(self, graph, cache, mode: str = "role"):
        """ Output ChronoGrapher -> Singleton Property 
        cache: since properties need to be common across all kgs,
        caching meaning of single properties """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res, mode=mode)

        data = []
        if cache:
            nb = max(int(x.split('#')[1]) for _, x in cache.items()) + 1
        else:
            nb = 1
        for _, row in res[["sub_event", "role_type", "role"]].drop_duplicates().iterrows():
            if str(row.role_type) in cache:  # property already created
                pred = cache[str(row.role_type)]
            else:
                pred = f"{self.prefixes['sem']}hasActor#{str(nb)}"
                nb += 1
                cache[str(row.role_type)] = pred
            data.extend([
                (row.sub_event, pred, row.role),
                (pred, f"{self.prefixes['rdfs']}subPropertyOf", f"{self.prefixes['sem']}hasActor"),
                (pred, f"{self.prefixes['sem']}hasRole", row.role_type)
            ])
        sub_graphs.append(pd.DataFrame(data, columns=self.triples_col).drop_duplicates())
        return self.get_output_from_subgraph(sg=sub_graphs), cache

    def to_simple_rdf_reification(self, graph, statement_nb, mode: str = "role"):
        """ Output ChronoGrapher -> Reification
        statement_nb: to ensure non-overlapping IDs across events """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res, mode=mode)

        data = []
        for _, row in res[["sub_event", "role_type", "role"]].drop_duplicates().iterrows():
            statement_node = f"{self.prefixes['ex']}Statement#{str(statement_nb)}"
            statement_nb += 1
            data.extend([
                (statement_node, f"{self.prefixes['ex']}subject", row.sub_event),
                (statement_node, f"{self.prefixes['ex']}predicate",
                    f"{self.prefixes['sem']}hasActor"),
                (statement_node, f"{self.prefixes['ex']}object", row.role),
                (statement_node, f"{self.prefixes['sem']}hasRole", row.role_type),
            ])
        sub_graphs.append(pd.DataFrame(data, columns=self.triples_col).drop_duplicates())
        return self.get_output_from_subgraph(sg=sub_graphs), statement_nb

    def to_hypergraph_bn(self, graph, mode: str = "role"):
        """ Output ChronoGrapher -> hypergraph with bn """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res, mode=mode)
        sub_graphs = self.get_output_from_subgraph(sg=sub_graphs)[self.triples_col]
        sub_graphs.columns = range(sub_graphs.shape[1])

        data = []
        for _, row in res[["sub_event", "role_type", "role"]].drop_duplicates().iterrows():
            data.append(
                (f"<{row.sub_event}>", f"<{self.prefixes['sem']}hasActor>", f"<{row.role}>", 
                    f"<{self.prefixes['sem']}hasRole>", f"<{row.role_type}>")
            )
        df = pd.DataFrame(data=data, columns=range(5))
        return pd.concat([sub_graphs, df], ignore_index=True)

    def to_hyper_relational_rdf_star(self, graph, mode: str = "role"):
        """ Output ChronoGrapher -> hyper-relational with rdf-star """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res, mode=mode)
        sub_graphs = self.get_output_from_subgraph(sg=sub_graphs)
        sub_graphs.columns = range(sub_graphs.shape[1])

        data = []
        for _, row in res[["sub_event", "role_type", "role"]].drop_duplicates().iterrows():
            data.append(
                ("<<", f"<{row.sub_event}>", f"<{self.prefixes['sem']}hasActor>", f"<{row.role}>", ">>",
                    f"<{self.prefixes['sem']}hasRole>", f"<{row.role_type}>", ".")
            )
        df = pd.DataFrame(data=data, columns=range(8))
        return pd.concat([sub_graphs, df], ignore_index=True)



if __name__ == '__main__':
    FILE_P = "frame_ng.nt"
    FILE_P = "kg_test/cg_output.nt"
    GRAPH = Graph()
    GRAPH.parse(FILE_P)
    DF = read_nt(file_p=FILE_P)
    CONVERTER = KGRepresentationsConverter()

    # Simple RDF
    RES = CONVERTER.to_simple_rdf_prop(graph=GRAPH)
    ADD_TEXT = False
    if ADD_TEXT:
        RES = CONVERTER.add_text(df=DF, output=RES)
    print(RES)
    print(RES.columns)

    # Singleton Property
    RES, CACHE = CONVERTER.to_simple_rdf_sp(graph=GRAPH, cache={})
    ADD_TEXT = True
    if ADD_TEXT:
        RES = CONVERTER.add_text(df=DF, output=RES)
    print(RES)
    RES.to_csv("test.csv", header=None, index=False, sep=" ")
    print(CACHE)

    # Reification
    RES, NB = CONVERTER.to_simple_rdf_reification(graph=GRAPH, statement_nb=1)
    ADD_TEXT = False
    if ADD_TEXT:
        RES = CONVERTER.add_text(df=DF, output=RES)
    print(RES)

    # Hypergraph (~blank nodes)
    RES = CONVERTER.to_hypergraph_bn(graph=GRAPH)
    print(RES)

    # RDF-star
    RES = CONVERTER.to_hyper_relational_rdf_star(graph=GRAPH)
    print(RES)
