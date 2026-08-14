"""Stateless SSU authority policies."""


VALID_POLICIES = {'baseline', 'audit_only', 'hard', 'soft', 'oracle_gt'}
COUNTERFACTUAL_AUDIT_POLICIES = {'audit_only', 'hard', 'soft'}


def counterfactual_auditing_active(policy, enabled=True):
    """Whether this policy actually requires CARF counterfactual reruns."""
    if policy not in VALID_POLICIES:
        raise ValueError('unknown CARF policy: %s' % policy)
    return bool(enabled and policy in COUNTERFACTUAL_AUDIT_POLICIES)


def authority_for_policy(policy, audit_result=None, soft_power=1.0,
                         identity_contamination=None, enabled=True):
    """Return q in [0, 1] without changing the accepted assignment.

    Missing rollback history and unknown oracle labels always fall back to the
    exact baseline authority q=1.
    """
    if policy not in VALID_POLICIES:
        raise ValueError('unknown CARF policy: %s' % policy)
    if not enabled or policy in {'baseline', 'audit_only'}:
        return 1.0

    if policy == 'oracle_gt':
        if identity_contamination is True:
            return 0.0
        return 1.0

    if audit_result is None or not audit_result.auditable:
        return 1.0

    survival = float(audit_result.survival)
    if policy == 'hard':
        return 1.0 if survival == 1.0 else 0.0

    soft_power = float(soft_power)
    if soft_power <= 0.0:
        raise ValueError('soft_power must be positive')
    return survival ** soft_power
