from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CASE_STUDIES = ("case_study_1", "case_study_2", "case_study_3")
OUTPUT_DIR = DATA_ROOT / "r2r_ranking_results"
LONG_OUTPUT_PATH = OUTPUT_DIR / "session_method_rankings_long.csv"
WIDE_OUTPUT_PATH = OUTPUT_DIR / "session_method_rankings_wide.csv"
COPELAND_ALPHA = 0.33
RATING_LEVELS = tuple(range(1, 6))
METHOD_COLUMNS = ["R2R_copeland", "R2R_borda", "Copeland", "Borda", "Mean", "Median"]
OUTPUT_COLUMNS = [
    "case_study",
    "session_id",
    "group_local",
    "group_global",
    "method",
    "score",
    "rank",
]


def rating_session_id(path: Path) -> str:
    """Return the session id encoded by a rating CSV path."""
    stem = path.stem
    if stem.endswith("_rate"):
        return stem[: -len("_rate")]
    if stem.isdigit():
        return stem
    raise ValueError(f"Unexpected rating filename format: {path}")


def ranking_session_id(path: Path) -> str:
    """Return the session id encoded by a ranking CSV path."""
    stem = path.stem
    if stem.endswith("_rank"):
        return stem[: -len("_rank")]
    raise ValueError(f"Unexpected ranking filename format: {path}")


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    """Raise a clear error when a CSV is missing required columns."""
    missing = columns.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_columns}")


def parse_ranking(value: object, path: Path) -> list[int]:
    """Parse the list_rank2 value used by the reference implementation."""
    if isinstance(value, list):
        ranking = value
    else:
        try:
            ranking = ast.literal_eval(str(value))
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse ranking in {path}: {value!r}") from exc

    if not isinstance(ranking, list):
        raise ValueError(f"Ranking value in {path} is not a list: {value!r}")

    try:
        return [int(label) for label in ranking]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ranking value in {path} contains a non-integer: {value!r}") from exc


def discover_session_files(case_study_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Find matching rating and ranking files for one case-study directory."""
    ratings_dir = case_study_dir / "ratings"
    rankings_dir = case_study_dir / "rankings"

    if not ratings_dir.is_dir():
        raise ValueError(f"Missing ratings directory: {ratings_dir}")
    if not rankings_dir.is_dir():
        raise ValueError(f"Missing rankings directory: {rankings_dir}")

    rating_files = {rating_session_id(path): path for path in ratings_dir.glob("*.csv")}
    ranking_files = {ranking_session_id(path): path for path in rankings_dir.glob("*.csv")}

    missing_ratings = sorted(set(ranking_files).difference(rating_files), key=int)
    missing_rankings = sorted(set(rating_files).difference(ranking_files), key=int)
    if missing_ratings:
        missing = ", ".join(missing_ratings)
        raise ValueError(f"{case_study_dir.name} ranking session(s) missing rating file: {missing}")
    if missing_rankings:
        missing = ", ".join(missing_rankings)
        raise ValueError(f"{case_study_dir.name} rating session(s) missing ranking file: {missing}")

    return {
        session_id: (rating_files[session_id], ranking_files[session_id])
        for session_id in sorted(rating_files, key=int)
    }


def grades_per_group(ratings: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized grade distribution, mean, raw median, and floored median."""
    grade_counts = ratings.groupby(["group_number", "rate"])["username"].count().unstack()
    normalized = grade_counts.div(grade_counts.sum(axis=1), axis=0)
    normalized = normalized.reindex(columns=RATING_LEVELS, fill_value=0).fillna(0)
    normalized = normalized.reset_index()
    normalized.columns = ["group_number", *[str(level) for level in RATING_LEVELS]]

    summary = (
        ratings.groupby("group_number")["rate"]
        .agg(Mean="mean", Median_raw="median")
        .reset_index()
    )
    stats = normalized.merge(summary, on="group_number")
    stats["Median"] = np.floor(stats["Median_raw"])
    return stats


def count_pairwise_wins(rankings: pd.Series, labels: list[int], path: Path) -> dict[tuple[int, int], int]:
    """Count pairwise wins for each label pair using the original ranking comparison rule."""
    parsed_rankings = [parse_ranking(ranking, path) for ranking in rankings]
    counts: dict[tuple[int, int], int] = {}

    for index, label1 in enumerate(labels):
        for label2 in labels[index + 1 :]:
            counts[(label1, label2)] = 0
            counts[(label2, label1)] = 0

            for ranking in parsed_rankings:
                label1_position = ranking.index(label1)
                label2_position = ranking.index(label2)
                if label1_position < label2_position:
                    counts[(label1, label2)] += 1
                elif label2_position < label1_position:
                    counts[(label2, label1)] += 1
                else:
                    raise ValueError(
                        f"Labels {label1} and {label2} have the same position in {path}"
                    )

    return counts


def copeland_method(rankings: pd.Series, labels: list[int], alpha: float, path: Path) -> dict[int, float]:
    """Compute the Copeland score exactly as in the reference implementation."""
    counts = count_pairwise_wins(rankings, labels, path)
    scores: dict[int, float] = {}

    for label in labels:
        label_wins = []
        for other_label in labels:
            if label == other_label:
                continue

            wins = counts[(label, other_label)]
            half_voters = len(rankings) / 2
            if float(wins) > float(half_voters):
                label_wins.append(1)
            elif float(wins) == float(half_voters):
                label_wins.append(alpha)
            else:
                label_wins.append(0)

        scores[label] = float(sum(label_wins))

    return scores


def borda_method(rankings: pd.Series, labels: list[int], path: Path) -> dict[int, float]:
    """Compute the Borda-style pairwise win score from the reference implementation."""
    counts = count_pairwise_wins(rankings, labels, path)
    scores: dict[int, float] = {}

    for label in labels:
        scores[label] = float(
            sum(counts[(label, other_label)] for other_label in labels if label != other_label)
        )

    return scores


def score_tied_medians(
    stats: pd.DataFrame,
    rankings: pd.Series,
    method: str,
    alpha: float,
    path: Path,
) -> dict[int, float]:
    """Apply Copeland or Borda only inside groups tied by floored median."""
    scores: dict[int, float] = {}

    for median_value in stats["Median"].unique():
        tied_labels = stats.loc[stats["Median"] == median_value, "group_number"].astype(int).tolist()

        if len(tied_labels) == 1:
            scores[tied_labels[0]] = 0
        elif method == "copeland":
            scores.update(copeland_method(rankings, tied_labels, alpha, path))
        elif method == "borda":
            scores.update(borda_method(rankings, tied_labels, path))
        else:
            raise ValueError(f"Unknown tie-breaking method: {method}")

    if method == "copeland":
        normalization_factor = len(rankings)
    elif method == "borda":
        normalization_factor = len(rankings) * max(RATING_LEVELS)
    else:
        raise ValueError(f"Unknown tie-breaking method: {method}")

    return {label: score / normalization_factor for label, score in scores.items()}


def compute_method_scores(
    ratings: pd.DataFrame,
    rankings: pd.DataFrame,
    rating_path: Path,
    ranking_path: Path,
) -> pd.DataFrame:
    """Compute all requested project scores for one session."""
    require_columns(ratings, {"username", "group_number", "rate"}, rating_path)
    require_columns(rankings, {"list_rank2"}, ranking_path)

    stats = grades_per_group(ratings)
    labels = stats["group_number"].astype(int).tolist()
    ranking_lists = rankings["list_rank2"]

    r2r_copeland = score_tied_medians(stats, ranking_lists, "copeland", COPELAND_ALPHA, ranking_path)
    r2r_borda = score_tied_medians(stats, ranking_lists, "borda", 0, ranking_path)
    copeland = copeland_method(ranking_lists, labels, COPELAND_ALPHA, ranking_path)
    borda = borda_method(ranking_lists, labels, ranking_path)

    scores = stats[["group_number", "Mean", "Median_raw"]].copy()
    scores["group_number"] = scores["group_number"].astype(int)
    scores["R2R_copeland"] = scores["Median_raw"] + scores["group_number"].map(r2r_copeland)
    scores["R2R_borda"] = scores["Median_raw"] + scores["group_number"].map(r2r_borda)
    scores["Copeland"] = scores["group_number"].map(copeland)
    scores["Borda"] = scores["group_number"].map(borda)
    scores["Mean"] = scores["Mean"]
    scores["Median"] = scores["Median_raw"]
    return scores[["group_number", "R2R_copeland", "R2R_borda", "Copeland", "Borda", "Mean", "Median"]]


def to_long_rankings(
    case_study: str,
    session_id: str,
    scores: pd.DataFrame,
) -> list[dict[str, object]]:
    """Convert one session's wide scores to the requested long ranking rows."""
    rank_frame = scores[METHOD_COLUMNS].rank(ascending=False, method="min").astype(int)
    rows: list[dict[str, object]] = []

    for score_row, rank_row in zip(scores.itertuples(index=False), rank_frame.itertuples(index=False)):
        group_local = int(score_row.group_number)
        group_global = f"{case_study}_{session_id}_{group_local}"

        for method in METHOD_COLUMNS:
            rows.append(
                {
                    "case_study": case_study,
                    "session_id": int(session_id),
                    "group_local": group_local,
                    "group_global": group_global,
                    "method": method,
                    "score": getattr(score_row, method),
                    "rank": int(getattr(rank_row, method)),
                }
            )

    return rows


def build_session_method_rankings() -> tuple[pd.DataFrame, int, int]:
    """Process every configured case-study session and return the output data."""
    rows: list[dict[str, object]] = []
    sessions_processed = 0
    projects_processed = 0

    for case_study in CASE_STUDIES:
        session_files = discover_session_files(DATA_ROOT / case_study)

        for session_id, (rating_path, ranking_path) in session_files.items():
            ratings = pd.read_csv(rating_path)
            rankings = pd.read_csv(ranking_path)
            require_columns(ratings, {"username", "group_number", "rate"}, rating_path)
            require_columns(rankings, {"list_rank2"}, ranking_path)

            scores = compute_method_scores(ratings, rankings, rating_path, ranking_path)
            rows.extend(to_long_rankings(case_study, session_id, scores))
            sessions_processed += 1
            projects_processed += len(scores)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), sessions_processed, projects_processed


def write_wide_method_comparison(rankings: pd.DataFrame) -> None:
    """Write one row per session/group with each method's rank in its own column."""
    wide = (
        rankings.pivot(
            index=["case_study", "session_id", "group_local", "group_global"],
            columns="method",
            values="rank",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide = wide[["case_study", "session_id", "group_local", "group_global", *METHOD_COLUMNS]]
    wide = wide.sort_values(["session_id", "group_local"])
    wide.to_csv(WIDE_OUTPUT_PATH, index=False)


def main() -> None:
    """Write all session-level method-ranking outputs."""
    output, sessions_processed, projects_processed = build_session_method_rankings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(LONG_OUTPUT_PATH, index=False)
    write_wide_method_comparison(output)

    print(f"sessions_processed: {sessions_processed}")
    print(f"projects_processed: {projects_processed}")
    print(f"rows_written: {len(output)}")
    print(f"wrote: {LONG_OUTPUT_PATH}")
    print(f"wrote: {WIDE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
