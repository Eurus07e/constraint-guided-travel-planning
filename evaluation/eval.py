"""Compatibility entry point for full, indexed benchmark submissions."""
from evaluation.protocol import evaluate_file, official_scores


def eval_score(set_type, file_path):
    result = evaluate_file(file_path, set_type, require_complete=True)
    return official_scores(result['summary']), {'protocol':result['protocol'], 'cases':result['cases']}
