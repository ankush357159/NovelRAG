import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class EvalRow:
    query: str
    novel: str
    expected_chapter: str


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def _rank_of_expected(results: list[dict], expected_chapter: str) -> int | None:
    for i, row in enumerate(results, start=1):
        chapter = row.get("metadata", {}).get("chapter_label")
        if chapter == expected_chapter:
            return i
    return None


def _dense_search(base_url: str, row: EvalRow, k: int) -> list[dict]:
    q = urllib.parse.quote(row.query)
    n = urllib.parse.quote(row.novel)
    url = f"{base_url}/vectorstore/v1/search?query={q}&k={k}&novel={n}"
    payload = _get_json(url)
    return payload.get("results", [])


def _hybrid_search(base_url: str, row: EvalRow, k: int, fetch_k: int, rrf_k: int) -> list[dict]:
    q = urllib.parse.quote(row.query)
    n = urllib.parse.quote(row.novel)
    url = (
        f"{base_url}/vectorstore/v1/hybrid-search?query={q}&k={k}"
        f"&fetch_k={fetch_k}&rrf_k={rrf_k}&novel={n}"
    )
    payload = _get_json(url)
    return payload.get("results", [])


def _compute_metrics(ranks: list[int | None], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    total = len(ranks)
    hit = {}
    for k in k_values:
        hit[f"hit@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / total, 4)

    mrr = 0.0
    for r in ranks:
        if r is not None:
            mrr += 1.0 / r
    mrr = round(mrr / total, 4)

    return {"count": total, **hit, "mrr": mrr}


def _load_eval_rows(path: str) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                EvalRow(
                    query=obj["query"],
                    novel=obj["novel"],
                    expected_chapter=obj["expected_chapter"],
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense vs hybrid retrieval rank metrics.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--queries", default="app/api/retrieval_eval_seed.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fetch-k", type=int, default=120)
    parser.add_argument("--rrf-k", type=int, default=40)
    args = parser.parse_args()

    rows = _load_eval_rows(args.queries)

    dense_ranks: list[int | None] = []
    hybrid_ranks: list[int | None] = []

    for row in rows:
        dense_results = _dense_search(args.base_url, row, args.k)
        hybrid_results = _hybrid_search(args.base_url, row, args.k, args.fetch_k, args.rrf_k)

        dense_rank = _rank_of_expected(dense_results, row.expected_chapter)
        hybrid_rank = _rank_of_expected(hybrid_results, row.expected_chapter)

        dense_ranks.append(dense_rank)
        hybrid_ranks.append(hybrid_rank)

        print("-")
        print(f"query: {row.query}")
        print(f"expected_chapter: {row.expected_chapter}")
        print(f"dense_rank: {dense_rank}")
        print(f"hybrid_rank: {hybrid_rank}")

    print("\n=== summary ===")
    print("dense:", json.dumps(_compute_metrics(dense_ranks), indent=2))
    print("hybrid:", json.dumps(_compute_metrics(hybrid_ranks), indent=2))


if __name__ == "__main__":
    main()
