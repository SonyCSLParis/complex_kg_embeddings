# -*- coding: utf-8 -*-
"""
Converting the output of ChronoGrapher into different formats
"""
import json
from rdflib import Graph
import pandas as pd

def get_columns_from_query(query):
    """ column names from sparql query """
    res = query.split(" WHERE {")[0].split("?")
    res = [x.strip() for x in res if 'select' not in x.lower()]
    return res


def read_nt(file_p):
    """ Read .nt file as a csv """
    df = pd.read_csv(file_p, sep=' ', header=None)
    df.columns = ["subject", "predicate", "object", "."]
    return df


def query_return_csv(graph, query):
    """ (SELECT) query to be returned as .csv """
    res = graph.query(query)
    res = pd.DataFrame(res)
    res.columns = get_columns_from_query(query=query)
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
            ?event ex:abstract ?abstract .
            ?abstract nif:sentence ?sentence .
            ?sub_event wsj:fromDocument ?sentence ;
                    a wsj:CorpusEntry ;
                    wsj:onFrame ?frame ;
                    wsj:withmappedrole ?role .
            ?role wsj:withfnfe ?role_type .
            OPTIONAL {?role skos:related ?entity}
            }
            """
        }
        self.queries = {k: self.prefixes_query + "\n" + val for k, val in self.queries.items()}
        self.triples_col = ["subject", "predicate", "object"]

    def link_two_cols(self, df, cols, pred):
        """ Link two cols in a df with a predicate
        cols[0] will be the subject and cols[1] the object """
        res = df[cols].drop_duplicates()
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

    def init_sub_graphs(self, df):
        """ Sub graphs common to all representation types """
        sub_graphs = []
        sub_graphs.append(self.link_two_cols(
            df=df, cols=["sub_event", "event"], pred=f"{self.prefixes['sem']}subEventOf"))
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

    def to_simple_rdf_with_properties(self, graph):
        """ Output ChronoGrapher -> Simple RDF (roles with properties) """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])

        sub_graphs = self.init_sub_graphs(df=res)
        sub_graphs.append(
            res[["sub_event", "role_type", "role"]].drop_duplicates().rename(
                columns={"sub_event": "subject", "role_type": "predicate", "role": "object"}))

        return self.get_output_from_subgraph(sg=sub_graphs)

    def to_simple_rdf_with_sp(self, graph, cache):
        """ Output ChronoGrapher -> Singleton Property 
        cache: since properties need to be common across all kgs,
        caching meaning of single properties """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res)

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

    def to_simple_rdf_with_reification(self, graph, statement_nb):
        """ Output ChronoGrapher -> Reification
        statement_nb: to ensure non-overlapping IDs across events """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res)

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

    def to_hypergraph_with_bn(self, graph):
        """ Output ChronoGrapher -> hypergraph with bn """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res)
        sub_graphs = self.get_output_from_subgraph(sg=sub_graphs)[self.triples_col]
        sub_graphs.columns = range(sub_graphs.shape[1])

        data = []
        for _, row in res[["sub_event", "role_type", "role"]].drop_duplicates().iterrows():
            data.append(
                (f"<{row.sub_event}>", f"<{self.prefixes['sem']}hasActor>", f"<{row.role}>", 
                    f"<{self.prefixes['sem']}hasRole>", f"<{row.role_type}>")
            )
        df = pd.DataFrame(data=data, columns=range(len(data[0])))
        return pd.concat([sub_graphs, df], ignore_index=True)

    def to_hyper_relational_with_rdf_star(self, graph):
        """ Output ChronoGrapher -> hyper-relational with rdf-star """
        res = query_return_csv(graph=graph, query=self.queries["simple_rdf_with_properties"])
        sub_graphs = self.init_sub_graphs(df=res)
        sub_graphs = self.get_output_from_subgraph(sg=sub_graphs)
        sub_graphs.columns = range(sub_graphs.shape[1])

        data = []
        for _, row in res[["sub_event", "role_type", "role"]].drop_duplicates().iterrows():
            data.append(
                ("<<", f"<{row.sub_event}>", f"<{self.prefixes['sem']}hasActor>", f"<{row.role}>", ">>",
                    f"<{self.prefixes['sem']}hasRole>", f"<{row.role_type}>", ".")
            )
        df = pd.DataFrame(data=data, columns=range(len(data[0])))
        return pd.concat([sub_graphs, df], ignore_index=True)



if __name__ == '__main__':
    FILE_P = "frame_ng.nt"
    FILE_P = "ChronoGrapher/kg_test/cg_output.nt"
    GRAPH = Graph()
    GRAPH.parse(FILE_P)
    DF = read_nt(file_p=FILE_P)
    CONVERTER = KGRepresentationsConverter()

    # RES = CONVERTER.to_simple_rdf_with_properties(graph=GRAPH)
    # ADD_TEXT = False
    # if ADD_TEXT:
    #     RES = CONVERTER.add_text(df=DF, output=RES)
    # print(RES)
    # print(RES.columns)
    # RES.to_csv("test.csv", header=None, index=False, sep=" ")

    # RES, CACHE = CONVERTER.to_simple_rdf_with_sp(graph=GRAPH, cache={})
    # ADD_TEXT = True
    # if ADD_TEXT:
    #     RES = CONVERTER.add_text(df=DF, output=RES)
    # print(RES)
    # RES.to_csv("test.csv", header=None, index=False, sep=" ")
    # print(CACHE)
    # with open("logs.json", 'w', encoding='utf-8') as f:
    #     json.dump(CACHE, f, indent=4)

    # RES, NB = CONVERTER.to_simple_rdf_with_reification(graph=GRAPH, statement_nb=1)
    # ADD_TEXT = False
    # if ADD_TEXT:
    #     RES = CONVERTER.add_text(df=DF, output=RES)
    # print(RES)
    # RES.to_csv("test.csv", header=None, index=False, sep=" ")
    # print(NB)

    # RES = CONVERTER.to_hypergraph_with_bn(graph=GRAPH)
    # print(RES)
    # RES.to_csv("output.txt", sep=" ", header=False, index=False, na_rep="")

    RES = CONVERTER.to_hyper_relational_with_rdf_star(graph=GRAPH)
    print(RES)
    RES.to_csv("output.txt", sep=" ", header=False, index=False, na_rep="")
