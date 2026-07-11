"""Construcción y análisis del grafo bipartito cliente-comercio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(slots=True)
class GraphOutputs:
    graph: nx.DiGraph
    graph_base: pd.DataFrame
    edge_table: pd.DataFrame
    components: pd.DataFrame
    degrees: pd.DataFrame
    centrality: pd.DataFrame
    merchant_ranking: pd.DataFrame
    ranking_comparison: pd.DataFrame


def add_anonymous_node_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Crea identificadores estables sin exponer PII directa."""
    result = data.copy()
    proxy_cols = [
        col
        for col in (
            "tarjeta_ultimos4",
            "estado",
            "codigo_postal",
            "fecha_nacimiento",
        )
        if col in result
    ]

    if proxy_cols:
        raw_proxy = (
            result[proxy_cols]
            .fillna("desconocido")
            .astype(str)
            .agg("|".join, axis=1)
        )
    else:
        raw_proxy = result.index.astype(str)

    result["cliente_id"] = (
        "cliente_"
        + pd.util.hash_pandas_object(raw_proxy, index=False)
        .astype("uint64")
        .astype(str)
        .str[-10:]
    )

    merchant = result["comercio"].fillna("comercio_desconocido").astype(str)
    result["comercio_id"] = (
        "comercio_"
        + pd.util.hash_pandas_object(merchant, index=False)
        .astype("uint64")
        .astype(str)
        .str[-10:]
    )
    return result


def build_graph(candidate_pool: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """Construye un grafo dirigido cliente -> comercio."""
    required = {"comercio", "monto", "score_modelo_fuerte"}
    missing = required.difference(candidate_pool.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el grafo: {sorted(missing)}")

    base = candidate_pool.copy()
    if "fraude_real" not in base:
        base["fraude_real"] = base.get("es_fraude", 0)

    base["monto"] = pd.to_numeric(base["monto"], errors="coerce").fillna(0.0)
    base["score_modelo_fuerte"] = pd.to_numeric(
        base["score_modelo_fuerte"], errors="coerce"
    ).fillna(0.0)
    base["fraude_real"] = pd.to_numeric(
        base["fraude_real"], errors="coerce"
    ).fillna(0).astype(int)
    base = add_anonymous_node_ids(base)

    edges = (
        base.groupby(["cliente_id", "comercio_id"], as_index=False)
        .agg(
            n_transacciones=("monto", "size"),
            monto_total=("monto", "sum"),
            monto_promedio=("monto", "mean"),
            fraudes=("fraude_real", "sum"),
            tasa_fraude=("fraude_real", "mean"),
            score_modelo_promedio=("score_modelo_fuerte", "mean"),
        )
    )
    edges["peso"] = edges["n_transacciones"].astype(float)
    edges["peso_riesgo"] = edges["n_transacciones"] * (
        1 + edges["score_modelo_promedio"]
    )

    graph = nx.DiGraph()
    for node in base["cliente_id"].drop_duplicates():
        graph.add_node(node, tipo_nodo="cliente")

    merchant_columns = ["comercio_id", "comercio"]
    if "categoria_comercio" in base:
        merchant_columns.append("categoria_comercio")
    merchants = base[merchant_columns].drop_duplicates("comercio_id")
    for row in merchants.itertuples(index=False):
        attrs = {
            "tipo_nodo": "comercio",
            "nombre_comercio": row.comercio,
        }
        if hasattr(row, "categoria_comercio"):
            attrs["categoria_comercio"] = row.categoria_comercio
        graph.add_node(row.comercio_id, **attrs)

    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.cliente_id,
            row.comercio_id,
            weight=float(row.peso),
            n_transacciones=int(row.n_transacciones),
            monto_total=float(row.monto_total),
            monto_promedio=float(row.monto_promedio),
            fraudes=int(row.fraudes),
            tasa_fraude=float(row.tasa_fraude),
            score_modelo_promedio=float(row.score_modelo_promedio),
            peso_riesgo=float(row.peso_riesgo),
        )
    return graph, base, edges


def connected_components_report(graph: nx.DiGraph) -> pd.DataFrame:
    """Resume componentes débilmente conectados."""
    rows = []
    components = sorted(
        nx.weakly_connected_components(graph), key=len, reverse=True
    )
    for component_id, nodes in enumerate(components, start=1):
        subgraph = graph.subgraph(nodes)
        rows.append(
            {
                "componente": component_id,
                "n_nodos": subgraph.number_of_nodes(),
                "n_aristas": subgraph.number_of_edges(),
                "n_clientes": sum(
                    data.get("tipo_nodo") == "cliente"
                    for _, data in subgraph.nodes(data=True)
                ),
                "n_comercios": sum(
                    data.get("tipo_nodo") == "comercio"
                    for _, data in subgraph.nodes(data=True)
                ),
            }
        )
    return pd.DataFrame(rows)


def degree_report(graph: nx.DiGraph) -> pd.DataFrame:
    """Calcula grados y grados ponderados por nodo."""
    rows = []
    for node, attrs in graph.nodes(data=True):
        rows.append(
            {
                "node_id": node,
                "tipo_nodo": attrs.get("tipo_nodo"),
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "degree": graph.degree(node),
                "weighted_in_degree": graph.in_degree(node, weight="weight"),
                "weighted_out_degree": graph.out_degree(node, weight="weight"),
                "weighted_degree": graph.degree(node, weight="weight"),
            }
        )
    return pd.DataFrame(rows)


def centrality_report(graph: nx.DiGraph) -> pd.DataFrame:
    """Calcula PageRank estructural y PageRank ponderado por riesgo."""
    pagerank = nx.pagerank(graph, weight="weight")
    pagerank_risk = nx.pagerank(graph, weight="peso_riesgo")
    rows = []
    for node, attrs in graph.nodes(data=True):
        rows.append(
            {
                "node_id": node,
                "tipo_nodo": attrs.get("tipo_nodo"),
                "nombre_comercio": attrs.get("nombre_comercio"),
                "categoria_comercio": attrs.get("categoria_comercio"),
                "pagerank": float(pagerank.get(node, 0.0)),
                "pagerank_riesgo": float(pagerank_risk.get(node, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def merchant_ranking_report(
    graph_base: pd.DataFrame,
    degrees: pd.DataFrame,
    centrality: pd.DataFrame,
) -> pd.DataFrame:
    """Compara centralidad, popularidad, monto y score predictivo."""
    merchant_summary = (
        graph_base.groupby(["comercio_id", "comercio"], as_index=False)
        .agg(
            n_transacciones=("monto", "size"),
            monto_total=("monto", "sum"),
            monto_promedio=("monto", "mean"),
            fraudes=("fraude_real", "sum"),
            tasa_fraude=("fraude_real", "mean"),
            score_modelo_promedio=("score_modelo_fuerte", "mean"),
        )
    )
    merchant_degrees = degrees.loc[
        degrees["tipo_nodo"].eq("comercio"),
        ["node_id", "weighted_in_degree"],
    ].rename(columns={"node_id": "comercio_id"})
    merchant_centrality = centrality.loc[
        centrality["tipo_nodo"].eq("comercio"),
        ["node_id", "pagerank", "pagerank_riesgo"],
    ].rename(columns={"node_id": "comercio_id"})

    ranking = (
        merchant_summary.merge(merchant_degrees, on="comercio_id", how="left")
        .merge(merchant_centrality, on="comercio_id", how="left")
    )
    ranking["rank_grafo_pagerank"] = ranking["pagerank"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_grafo_riesgo"] = ranking["pagerank_riesgo"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_popularidad"] = ranking["n_transacciones"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_monto"] = ranking["monto_total"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_modelo"] = ranking["score_modelo_promedio"].rank(
        ascending=False, method="min"
    ).astype(int)
    return ranking.sort_values("rank_grafo_pagerank")


def compare_rankings(ranking: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Calcula correlaciones de Spearman y solapamiento Top-N."""
    pairs = [
        (
            "PageRank del grafo vs Popularidad",
            "rank_grafo_pagerank",
            "rank_popularidad",
        ),
        (
            "PageRank del grafo vs Ranking por monto",
            "rank_grafo_pagerank",
            "rank_monto",
        ),
        (
            "PageRank del grafo vs Ranking del modelo",
            "rank_grafo_pagerank",
            "rank_modelo",
        ),
        (
            "PageRank riesgo vs Ranking del modelo",
            "rank_grafo_riesgo",
            "rank_modelo",
        ),
    ]
    rows = []
    for label, first, second in pairs:
        value = spearmanr(
            ranking[first], ranking[second], nan_policy="omit"
        ).statistic
        rows.append(
            {
                "comparacion": label,
                "metrica": "Spearman rank correlation",
                "valor": float(value),
            }
        )

    top_sets = {
        name: set(ranking.nsmallest(top_n, column)["comercio_id"])
        for name, column in {
            "PageRank": "rank_grafo_pagerank",
            "PageRank riesgo": "rank_grafo_riesgo",
            "Popularidad": "rank_popularidad",
            "Modelo": "rank_modelo",
        }.items()
    }
    for label, first, second in [
        ("Top 10 PageRank vs Top 10 Popularidad", "PageRank", "Popularidad"),
        ("Top 10 PageRank vs Top 10 Modelo", "PageRank", "Modelo"),
        (
            "Top 10 PageRank riesgo vs Top 10 Modelo",
            "PageRank riesgo",
            "Modelo",
        ),
    ]:
        rows.append(
            {
                "comparacion": label,
                "metrica": f"Overlap@{top_n}",
                "valor": len(top_sets[first] & top_sets[second]) / top_n,
            }
        )
    return pd.DataFrame(rows)


def analyze_graph(candidate_pool: pd.DataFrame) -> GraphOutputs:
    """Ejecuta el análisis completo de grafos."""
    graph, base, edges = build_graph(candidate_pool)
    components = connected_components_report(graph)
    degrees = degree_report(graph)
    centrality = centrality_report(graph)
    ranking = merchant_ranking_report(base, degrees, centrality)
    comparison = compare_rankings(ranking)
    return GraphOutputs(
        graph,
        base,
        edges,
        components,
        degrees,
        centrality,
        ranking,
        comparison,
    )


def save_graph_outputs(outputs: GraphOutputs, output_dir: str | Path) -> list[Path]:
    """Guarda los mismos tipos de artefactos usados en Semana 12."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "graph_edges_cliente_comercio_week12.csv",
        output_dir / "graph_connected_components_week12.csv",
        output_dir / "graph_degree_report_week12.csv",
        output_dir / "graph_centrality_report_week12.csv",
        output_dir / "graph_vs_model_ranking_week12.csv",
        output_dir / "ranking_comparison_report_week12.csv",
        output_dir / "grafo_cliente_comercio_week12.gexf",
    ]
    outputs.edge_table.to_csv(paths[0], index=False)
    outputs.components.to_csv(paths[1], index=False)
    outputs.degrees.to_csv(paths[2], index=False)
    outputs.centrality.to_csv(paths[3], index=False)
    outputs.merchant_ranking.to_csv(paths[4], index=False)
    outputs.ranking_comparison.to_csv(paths[5], index=False)
    nx.write_gexf(outputs.graph, paths[6])
    return paths
