import math


def hit_at_k(retrieved_pages, relevant_pages, k):
    return float(bool(set(retrieved_pages[:k]) & set(relevant_pages)))


def recall_at_k(retrieved_pages, relevant_pages, k):
    relevant = set(relevant_pages)
    if not relevant:
        return 0.0
    return len(set(retrieved_pages[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved_pages, relevant_pages, k):
    if k <= 0:
        return 0.0
    return len(set(retrieved_pages[:k]) & set(relevant_pages)) / k


def reciprocal_rank(retrieved_pages, relevant_pages):
    relevant = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_pages, relevance_grades, k):
    def discounted_gain(grades):
        return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))

    gains = [relevance_grades.get(page, 0) for page in retrieved_pages[:k]]
    ideal_gains = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal_dcg = discounted_gain(ideal_gains)
    if not ideal_dcg:
        return 0.0
    return discounted_gain(gains) / ideal_dcg


def forbidden_scope_pages(retrieved_pages, forbidden_pages, k):
    forbidden = set(forbidden_pages)
    return list(dict.fromkeys(page for page in retrieved_pages[:k] if page in forbidden))


def score_case(
    retrieved_pages,
    relevant_pages=None,
    relevance_grades=None,
    forbidden_pages=None,
    k=5,
):
    grades = relevance_grades or {}
    relevant = list(relevant_pages or (page for page, grade in grades.items() if grade > 0))
    forbidden = forbidden_scope_pages(retrieved_pages, forbidden_pages or [], k)
    return {
        "relevant_count": len(set(relevant)),
        "hit_at_k": hit_at_k(retrieved_pages, relevant, k),
        "recall_at_k": recall_at_k(retrieved_pages, relevant, k),
        "precision_at_k": precision_at_k(retrieved_pages, relevant, k),
        "reciprocal_rank": reciprocal_rank(retrieved_pages, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved_pages, grades or {page: 1 for page in relevant}, k),
        "forbidden_pages": forbidden,
        "has_forbidden_scope": bool(forbidden),
    }


def aggregate_summary(case_scores):
    scores = list(case_scores)
    count = len(scores)

    def average(key):
        return sum(score.get(key, 0.0) for score in scores) / count if count else 0.0

    return {
        "case_count": count,
        "hit_at_k": average("hit_at_k"),
        "recall_at_k": average("recall_at_k"),
        "precision_at_k": average("precision_at_k"),
        "mrr": average("reciprocal_rank"),
        "ndcg_at_k": average("ndcg_at_k"),
        "forbidden_scope_rate": average("has_forbidden_scope"),
        "average_latency_ms": average("latency_ms"),
    }
