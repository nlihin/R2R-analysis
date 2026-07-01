from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CASE_STUDIES = ("case_study_1", "case_study_2", "case_study_3")
OUTPUT_PATH = DATA_ROOT / "r2r_ranking_results" / "peer_rating_ranking_evaluations.csv"
OUTPUT_COLUMNS = [
    "case_study",
    "session_id",
    "uid_local",
    "group_local",
    "uid_global",
    "group_global",
    "score",
    "rank",
]


def rating_session_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rate"):
        return stem[: -len("_rate")]
    if stem.isdigit():
        return stem
    raise ValueError(f"Unexpected rating filename format: {path}")


def ranking_session_id(path: Path) -> str:
    stem = path.stem
    if not stem.endswith("_rank"):
        raise ValueError(f"Unexpected ranking filename format: {path}")
    return stem[: -len("_rank")]


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_columns}")


def parse_ranked_groups(value: object, path: Path, username: str) -> list[int]:
    try:
        ranked_groups = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"Could not parse list_rank2 for user {username!r} in {path}: {value!r}"
        ) from exc

    if not isinstance(ranked_groups, list):
        raise ValueError(
            f"list_rank2 for user {username!r} in {path} is not a list: {value!r}"
        )

    try:
        return [int(group) for group in ranked_groups]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"list_rank2 for user {username!r} in {path} contains a non-integer group: {value!r}"
        ) from exc


def build_ranking_lookup(ranking_df: pd.DataFrame, path: Path) -> dict[str, dict[int, int]]:
    require_columns(ranking_df, {"username", "list_rank2"}, path)

    lookup: dict[str, dict[int, int]] = {}
    for row in ranking_df.itertuples(index=False):
        username = str(row.username)
        ranked_groups = parse_ranked_groups(row.list_rank2, path, username)
        lookup[username] = {group: rank for rank, group in enumerate(ranked_groups, start=1)}

    return lookup


def discover_files(case_study_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    ratings_dir = case_study_dir / "ratings"
    rankings_dir = case_study_dir / "rankings"

    if not ratings_dir.is_dir():
        raise ValueError(f"Missing ratings directory: {ratings_dir}")
    if not rankings_dir.is_dir():
        raise ValueError(f"Missing rankings directory: {rankings_dir}")

    rating_files = {rating_session_id(path): path for path in ratings_dir.glob("*.csv")}
    ranking_files = {ranking_session_id(path): path for path in rankings_dir.glob("*.csv")}

    missing_rankings = sorted(set(rating_files).difference(ranking_files), key=int)
    if missing_rankings:
        missing = ", ".join(missing_rankings)
        raise ValueError(f"{case_study_dir.name} rating session(s) missing ranking file: {missing}")

    return rating_files, ranking_files


def build_dataset() -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    sessions_processed = 0

    for case_study in CASE_STUDIES:
        case_study_dir = DATA_ROOT / case_study
        rating_files, ranking_files = discover_files(case_study_dir)

        for session_id in sorted(rating_files, key=int):
            rating_path = rating_files[session_id]
            ranking_path = ranking_files[session_id]

            rating_df = pd.read_csv(rating_path)
            ranking_df = pd.read_csv(ranking_path)
            require_columns(rating_df, {"username", "group_number", "rate"}, rating_path)
            ranking_lookup = build_ranking_lookup(ranking_df, ranking_path)

            rated_users = set(rating_df["username"].astype(str))
            missing_users = sorted(rated_users.difference(ranking_lookup))
            if missing_users:
                users = ", ".join(missing_users)
                raise ValueError(
                    f"Rated user(s) in {rating_path} do not appear in matching ranking file "
                    f"{ranking_path}: {users}"
                )

            for row in rating_df.itertuples(index=False):
                uid_local = str(row.username)
                group_local = int(row.group_number)
                user_ranks = ranking_lookup[uid_local]

                if group_local not in user_ranks:
                    raise ValueError(
                        f"Rated group {group_local} for user {uid_local!r} in "
                        f"{rating_path} does not appear in that user's ranking in {ranking_path}"
                    )

                rows.append(
                    {
                        "case_study": case_study,
                        "session_id": int(session_id),
                        "uid_local": uid_local,
                        "group_local": group_local,
                        "uid_global": f"{case_study}_{session_id}_{uid_local}",
                        "group_global": f"{case_study}_{session_id}_{group_local}",
                        "score": row.rate,
                        "rank": user_ranks[group_local],
                    }
                )

            sessions_processed += 1

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), sessions_processed


def main() -> None:
    dataset, sessions_processed = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"sessions_processed: {sessions_processed}")
    print(f"evaluations: {len(dataset)}")
    print(dataset.head(10).to_string(index=False))
    print(f"wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
