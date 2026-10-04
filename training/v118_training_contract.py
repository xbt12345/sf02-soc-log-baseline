"""Checks required by the revised experiment, independent of model fitting."""


class ContractViolation(ValueError):
    pass


def require_terminal_checkpoint(completed_epochs, prediction_epoch, status):
    if status != 'fit_executed' or completed_epochs != 25 or prediction_epoch != 25:
        raise ContractViolation('Fixed-budget experiment requires a completed 25-epoch fit and epoch-25 predictions; earlier checkpoints are diagnostic only.')


def require_complete_population(expected_positions, scored_positions):
    expected = list(expected_positions)
    scored = list(scored_positions)
    if len(set(expected)) != len(expected) or len(set(scored)) != len(scored):
        raise ContractViolation('Population ledger contains duplicate rows.')
    if len(expected) != len(scored) or set(expected) != set(scored):
        raise ContractViolation('All registered held-out rows must be scored, including missing parameters and collateral records.')


def replication_allowed(primary):
    """Engineering conformance and smaller training loss cannot authorize replication."""
    required = ('complete_six_paired_fits', 'population_and_identity_verified',
                'ASA_M_errors_at_most_318', 'ASA_S_errors_at_most_2094',
                'ASA_total_errors_at_most_2170', 'M_S_not_worse_than_fresh_A',
                'two_folds_improve', 'S_outside_top3_not_worse',
                'full_each_class_recall_and_F1_not_worse')
    if any(k not in primary or not isinstance(primary[k], bool) for k in required):
        raise ContractViolation('Actual primary quality results are required; absent results cannot be assumed to pass.')
    return all(primary[k] for k in required)
