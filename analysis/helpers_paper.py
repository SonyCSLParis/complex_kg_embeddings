# -*- coding: utf-8 -*-
"""
Helpers for the paper
"""
import os
import pandas as pd

TEMPLATE = """
\\begin{table}[<position>]
    \\centering
    \\caption{<caption>}
    \\label{<label>}
    \\resizebox{<resize_col>\\columnwidth}{!}{
        \\setlength{\\tabcolsep}{8pt}
        \\renewcommand{\\arraystretch}{1.2}
        \\begin{tabular}{<alignment>}
            \\toprule
            <columns> \\\\
            <midrule_1>
            <sub_columns> 
            <midrule_2>
            <data> \\\\
            \\bottomrule
        \end{tabular}
    }
\end{table}
"""
ETA = ["prop", "subevent", "role", "causation"]

def get_start_end_multicol(multicol: list[int]) -> list[str]:
    curr_start, start, end = 1, [], []
    for x in multicol:
        start.append(curr_start)
        end.append(curr_start + x - 1)
        curr_start = end[-1] + 1
    return [f"{val}-{end[i]}" for i, val in enumerate(start)]

def check_alignment_data(columns: list[str], label: str, alignment: str, data: list[list[str]]):
    if len(columns) != len(alignment):
        raise ValueError(f"Params `{label}` and `alignment` should have the same length")
    if any(len(x) != len(columns) for x in data):
        raise ValueError("Each list in `data` must have the same length as `{label}`")

def check_args(columns: list[str], alignment: str,
               data: list[list[str]], sub_columns: list[str], multicol: list[int]):
    """ Checking if format of input param will match table """
    if (len(sub_columns) == 0) ^ (len(multicol) == 0):
        raise ValueError("Params `sub_columns` and `multicol` must be either both True or False")
    
    if sub_columns:
        check_alignment_data(columns=sub_columns, label="sub_columns", alignment=alignment, data=data)

        if len(multicol) != len(columns):
            raise ValueError("Params `multicol` and `columns` must have the same length")
        if sum(multicol) != len(sub_columns):
            raise ValueError("The sum of integers in `multicol` must be equal to the length of `sub_columns`")
    else:   # Only main columns
        check_alignment_data(columns=columns, label="sub_columns", alignment=alignment, data=data)
    return

def build_table(columns: list[str], alignment: str,
                caption: str, label: str, position: str,
                data: list[list[str]], sub_columns: list[str] = [], multicol: list[int] = [],
                resize_col: int = 1) -> str:
    """ 
    - `data`: list of list. Each element in the list corresponds to the set of values
    for one row of the table
    """
    check_args(columns=columns, alignment=alignment, data=data, sub_columns=sub_columns, multicol=multicol)
    if sub_columns: 
        columns = ["\\multicolumn{" + str(multicol[i]) + "}{c}{" + col + "}" for i, col in enumerate(columns)]

        start_end = get_start_end_multicol(multicol=multicol)
        midrule_1 = "\n\t".join(["\\cmidrule(lr){" + x + "}" for x in start_end])
        midrule_2 = midrule_1

        sub_columns = " & ".join(sub_columns) + "\\\\"

    else:
        midrule_1 = "\\midrule"
        midrule_2 = ""
        sub_columns = ""

    columns = " & ".join(columns)
    data = "\\\ \n".join([" & ".join(['{:,}'.format(y) if not isinstance(y, str) else y for y in x]) for x in data])

    table = TEMPLATE.replace("<position>", position).replace("<caption>", caption) \
        .replace("<label>", label).replace("<alignment>", alignment) \
            .replace("<columns>", columns).replace("<midrule_1>", midrule_1) \
            .replace("<sub_columns>", sub_columns).replace("<midrule_2>", midrule_2) \
            .replace("<data>", data).replace("<resize_col>", str(resize_col))
    return table

def get_sum_statements(df: pd.DataFrame):
    return df.groupby(ETA+["syntax"]).agg({"statements": "sum"}) \
        .pivot_table(index=ETA, columns="syntax", values="statements", fill_value="-") \
            .rename_axis(columns=None).reset_index()

def add_col_base(row, col_base):
    if isinstance(row[f"{col_base}_1"], str):
        row[col_base] = "-"
    else:
        row[col_base] = int((row[f"{col_base}_1"] + row[f"{col_base}_2"]) / 2)
    return row

def avg_value(df1, df2):
    merged = pd.merge(df1, df2, on=ETA, suffixes=('_1', '_2'))
    columns = ETA.copy()
    for col_base in [x for x in merged.columns if x not in ETA and x.endswith("_1")]:
        col_base = col_base.replace("_1", "")
        columns.append(col_base)
        merged = merged.apply(lambda row: add_col_base(row, col_base), axis=1)
    return merged[columns]


def build_table_statements(folder_data = "stats/"):
    df = pd.read_csv(os.path.join(folder_data, "metrics_ilp.csv"), index_col=0)
    res = get_sum_statements(df).reset_index(drop=True)
    res["hyper_relational_rdf_star"] = res["hyper_relational_rdf_star"].astype(int)

    df1 = pd.read_csv(os.path.join(folder_data, "metrics_simkgc.csv"), index_col=0)
    df1 = get_sum_statements(df1)
    df2 = pd.read_csv(os.path.join(folder_data, "metrics_ultra.csv"), index_col=0)
    df2 = get_sum_statements(df2)
    avg_df = avg_value(df1, df2).reset_index(drop=True)
    
    return pd.merge(avg_df,res,on=ETA, how="outer").fillna("-")

COL_TO_NAME = {
    "prop": "\\emph{prop}", "subevent": "\\emph{subevent}",
    "role": "\\emph{role}", "causation": "\\emph{causation}",
    "simple_rdf_reification": "\\emph{rdf-reif.}", "simple_rdf_sp": "\\emph{rdf-sp}", 
    "simple_rdf_prop": "\\emph{rdf-reg}", "hyper_relational_rdf_star": "\\emph{rdf-star}"
}


if __name__ == '__main__':
    df_statements = build_table_statements()
    print(df_statements.columns)
    latex_table = build_table(
        columns=[COL_TO_NAME[x] for x in df_statements.columns],
        alignment="r" * 8,
        caption="Number of statements for each dataset",
        label="tab:dataset-description-statements",
        position="h",
        data=df_statements.values,
        sub_columns=[],
        multicol=[],
        resize_col=1
    )
    print(f"{latex_table}\n=====")

    df_paper_metric_per_eta_ultra = pd.read_csv(os.path.join("stats", "paper_metric_per_eta_ultra.csv"), index_col=0)
    columns = ["prop", "subevent", "role", "causation"]
    for m in ["MRR", "H@1", "H@3", "H@10"]:
        columns.extend([f"{m}$\\uparrow$", f"$\\delta${m}"])
    latex_table = build_table(
        columns=columns,
        alignment="ccccrrrrrrrr",
        caption="Best metrics for each dataset run with ULTRA",
        label="tab:best-metric-eta-ultra",
        position="h",
        data=df_paper_metric_per_eta_ultra.values,
        sub_columns=[],
        multicol=[],
        resize_col=1
    )
    print(f"{latex_table}\n=====")

    df_paper_metric_per_eta_ilp = pd.read_csv(os.path.join("stats", "best_metric_per_eta_ilp.csv"), index_col=0)
    cols_orig = [
        "prop", "subevent", "role", "causation", 
        "AMR", "MRR",
        "H@1", "H@3", "H@10"]
    columns = [
        "prop", "subevent", "role", "causation", 
        "AMR$\\downarrow$", "MRR$\\uparrow$",
        "H@1$\\uparrow$", "H@3$\\uparrow$", "H@10$\\uparrow$"]
    latex_table = build_table(
        columns=columns,
        alignment="ccccrrrrr",
        caption="Best metrics for each dataset run with ILP",
        label="tab:best-metric-eta-ilp",
        position="h",
        data=df_paper_metric_per_eta_ilp[cols_orig].round(2).values,
        sub_columns=[],
        multicol=[],
        resize_col=1
    )
    print(f"{latex_table}\n=====")

    df_paper_metric_per_eta_simkgc = pd.read_csv(os.path.join("stats", "paper_metric_per_eta_simkgc.csv"), index_col=0)
    columns = ["prop", "subevent", "role", "causation"]
    for m in ["MRR", "H@1", "H@3", "H@10"]:
        columns.extend([f"{m}$\\uparrow$", f"$\\delta${m}"])
    latex_table = build_table(
        columns=columns,
        alignment="ccccrrrrrrrr",
        caption="Best metrics for each dataset run with SimKGC",
        label="tab:best-metric-eta-simkgc",
        position="h",
        data=df_paper_metric_per_eta_simkgc.values,
        sub_columns=[],
        multicol=[],
        resize_col=1
    )
    print(f"{latex_table}\n=====")

    df_syntax = pd.read_csv(os.path.join("stats", "syntax_mean.csv"), index_col=0)
    cols_orig = ["syntax", "MRR", "H@1", "H@3", "H@10"]
    cols_table = ["syntax", "MRR$\\uparrow$", "H@1$\\uparrow$", "H@3$\\uparrow$", "H@10$\\uparrow$"]
    df_syntax["syntax"] = df_syntax["syntax"].apply(lambda x: x.replace("_", "\\_"))
    latex_table = build_table(
        columns=cols_table,
        alignment="crrrr",
        caption="Averaged metrics for the different syntaxes",
        label="tab:avg-metric-syntax",
        position="h",
        data=df_syntax[cols_orig].round(2).values,
        sub_columns=[],
        multicol=[],
        resize_col=0.7
    )
    print(f"{latex_table}\n=====")

    columns = ["prop", "subevent", "role", "causation", "MRR"]
    df_3 = pd.read_csv(os.path.join("stats", "best_metric_per_eta_ultra.csv"), index_col=0)[columns].rename(columns={"MRR": "ULTRA"})
    for name, col in [("simkgc", "SimKGC"), ("ilp", "ILP")]:
        df = pd.read_csv(os.path.join("stats", f"best_metric_per_eta_{name}.csv"), index_col=0)[columns].rename(columns={"MRR": col})
        df_3 = pd.merge(df_3, df, on=columns[:-1], how="left")
    for col in ["ULTRA", "SimKGC", "ILP"]:
        df_3[col] = df_3[col].apply(lambda x: "-" if str(x) == "nan" else round(x, 2))
    df_3["sum"] = df_3["prop"] + df_3["subevent"] + df_3["causation"]
    latex_table = build_table(
        columns=["prop", "subevent", "role", "causation", "ULTRA", "SimKGC", "ILP"],
        alignment="ccccrrr",
        caption="MRR metrics comparison across three methods",
        label="tab:mrr-metrics-3-methods",
        position="h",
        data=df_3.sort_values(by="sum")[df_3.role==0][["prop", "subevent", "role", "causation", "ULTRA", "SimKGC", "ILP"]].values,
        sub_columns=[],
        multicol=[],
        resize_col=0.8
    )
    print(f"{latex_table}\n=====")