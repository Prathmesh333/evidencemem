# ruff: noqa: E501, F821 - executed inside the generated notebook namespace

from itertools import product


def hierarchical_paired_accuracy(
    correct_a_by_seed,
    correct_b_by_seed,
    *,
    seed,
    draws=5_000,
    batch_size=100,
):
    """Bootstrap paired accuracy differences across both seeds and examples."""
    correct_a = np.asarray(correct_a_by_seed, dtype=np.float64)
    correct_b = np.asarray(correct_b_by_seed, dtype=np.float64)
    if correct_a.shape != correct_b.shape or correct_a.ndim != 2:
        raise ValueError("Expected matching [seed, example] correctness matrices.")
    if correct_a.shape[0] < 2 or correct_a.shape[1] < 2:
        raise ValueError("Hierarchical inference needs at least two seeds and examples.")

    differences = correct_a - correct_b
    seed_deltas = differences.mean(axis=1)
    observed = float(seed_deltas.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(int(draws), dtype=np.float64)
    n_seeds, n_examples = differences.shape
    for start in range(0, int(draws), int(batch_size)):
        stop = min(start + int(batch_size), int(draws))
        count = stop - start
        sampled_seeds = rng.integers(0, n_seeds, size=(count, n_seeds))
        sampled_examples = rng.integers(
            0,
            n_examples,
            size=(count, n_seeds, n_examples),
        )
        sampled = differences[
            sampled_seeds[:, :, None],
            sampled_examples,
        ]
        samples[start:stop] = sampled.mean(axis=(1, 2))

    signs = np.asarray(list(product((-1.0, 1.0), repeat=n_seeds)))
    sign_flip_null = (signs * seed_deltas[None, :]).mean(axis=1)
    sign_flip_p = float(np.mean(np.abs(sign_flip_null) >= abs(observed) - 1e-15))
    centered_samples = samples - observed
    bootstrap_p = (
        np.count_nonzero(np.abs(centered_samples) >= abs(observed) - 1e-15) + 1
    ) / (len(samples) + 1)
    return {
        "delta": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_p_two_sided": float(bootstrap_p),
        "seed_sign_flip_p_two_sided": sign_flip_p,
        "seed_deltas": [float(value) for value in seed_deltas],
        "draws": int(draws),
        "n_seeds": int(n_seeds),
        "n_examples": int(n_examples),
    }


def evaluate_frozen_hypothesis(specification, result):
    margin = float(specification["margin"])
    if specification["alternative"] == "noninferiority":
        decision_boundary = -margin
    elif specification["alternative"] == "superiority":
        decision_boundary = margin
    else:
        raise ValueError(
            f"Unknown alternative {specification['alternative']!r}."
        )
    supported = bool(result["ci_low"] > decision_boundary)
    return {
        "decision_boundary": float(decision_boundary),
        "supported": supported,
        "decision_rule": (
            "supported iff the hierarchical 95% CI lower bound is greater than "
            f"{decision_boundary:+.4f}"
        ),
    }


if EVALUATION_STAGE != "confirmatory":
    raise RuntimeError("The frozen hypothesis cell may run only in confirmatory mode.")
if set(ENCODER_DATA) != {CONFIRMATORY_ENCODER_KEY}:
    raise RuntimeError("Confirmatory inference must contain exactly the frozen encoder.")

encoder_key = CONFIRMATORY_ENCODER_KEY
evaluation_labels = ENCODER_DATA[encoder_key][EVALUATION_SPLIT][1]
train_labels = ENCODER_DATA[encoder_key]["train"][1]
evidence_method = FROZEN_DEVELOPMENT_RECORD["method"]
hypothesis_rows = []

for index, specification in enumerate(
    FROZEN_DEVELOPMENT_RECORD["confirmatory_hypotheses"]
):
    baseline_method = specification["baseline"]
    correct_evidence = np.stack(
        [
            prediction_bank[(encoder_key, int(seed), evidence_method)]
            == evaluation_labels
            for seed in CFG.seeds
        ]
    )
    correct_baseline = np.stack(
        [
            prediction_bank[(encoder_key, int(seed), baseline_method)]
            == evaluation_labels
            for seed in CFG.seeds
        ]
    )
    result = hierarchical_paired_accuracy(
        correct_evidence,
        correct_baseline,
        seed=CFG.sample_seed + 10_000 + index,
    )
    decision = evaluate_frozen_hypothesis(specification, result)
    evidence_examples = int(CFG.default_budget * N_CLASSES)
    baseline_examples = (
        int(len(train_labels))
        if baseline_method == "Full kNN"
        else evidence_examples
    )
    hypothesis_rows.append(
        {
            "hypothesis_id": specification["id"],
            "role": specification["role"],
            "encoder_key": encoder_key,
            "method": evidence_method,
            "baseline": baseline_method,
            "alternative": specification["alternative"],
            "margin": float(specification["margin"]),
            "accuracy_delta": result["delta"],
            "accuracy_delta_pp": 100.0 * result["delta"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "ci_low_pp": 100.0 * result["ci_low"],
            "ci_high_pp": 100.0 * result["ci_high"],
            "bootstrap_p_two_sided": result["bootstrap_p_two_sided"],
            "seed_sign_flip_p_two_sided": result[
                "seed_sign_flip_p_two_sided"
            ],
            "seed_deltas": json.dumps(result["seed_deltas"]),
            "draws": result["draws"],
            "n_seeds": result["n_seeds"],
            "n_examples": result["n_examples"],
            "evidence_stored_examples": evidence_examples,
            "baseline_stored_examples": baseline_examples,
            "baseline_to_evidence_storage_ratio": (
                float(baseline_examples / evidence_examples)
            ),
            "decision_boundary": decision["decision_boundary"],
            "supported": decision["supported"],
            "decision_rule": decision["decision_rule"],
        }
    )

CONFIRMATORY_HYPOTHESES = pd.DataFrame(hypothesis_rows)
expected_hypotheses = {
    item["id"] for item in FROZEN_DEVELOPMENT_RECORD["confirmatory_hypotheses"]
}
if set(CONFIRMATORY_HYPOTHESES["hypothesis_id"]) != expected_hypotheses:
    raise AssertionError("The confirmatory hypothesis registry is incomplete.")
atomic_csv(
    CONFIRMATORY_HYPOTHESES,
    RUN_DIR / "confirmatory_hypotheses.csv",
)
atomic_json(
    RUN_DIR / "confirmatory_hypotheses.json",
    {
        "frozen_before_confirmatory_evaluation": True,
        "development_record": FROZEN_DEVELOPMENT_RECORD,
        "results": CONFIRMATORY_HYPOTHESES.to_dict(orient="records"),
    },
)
display(
    CONFIRMATORY_HYPOTHESES[
        [
            "hypothesis_id",
            "baseline",
            "accuracy_delta_pp",
            "ci_low_pp",
            "ci_high_pp",
            "seed_sign_flip_p_two_sided",
            "baseline_to_evidence_storage_ratio",
            "supported",
        ]
    ]
)
