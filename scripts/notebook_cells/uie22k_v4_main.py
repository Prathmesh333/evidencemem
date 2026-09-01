# ruff: noqa: B023, E501, F821 - executed inside the generated notebook namespace

classification_rows = []
tuning_rows = []
reliability_tuning_rows = []
prediction_bank = {}
primary_memories = {}
qualitative_bank = {}
selected_hyperparameters = {}

MAIN_METHODS = (
    "CLIP zero-shot",
    "Full kNN",
    "Linear probe",
    "Random memory fused",
    "Centroid fused",
    "KMeans medoids fused",
    "Facility selection (no reliability) fused",
    "EvidenceMem v2 selection fused",
    "EvidenceMem v4 global fusion",
    "EvidenceMem v4 class-conditional visual",
    "EvidenceMem v4 fixed fusion",
    "EvidenceMem v4 continuous fusion",
    "Tip-Adapter (matched cache)",
)


def linear_probe_scores(model, features):
    scores = np.asarray(model.decision_function(features))
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    return scores

for encoder_key, data in ENCODER_DATA.items():
    spec = data["spec"]
    resolution = int(spec["resolution"])
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    evaluation_x, evaluation_y = data[EVALUATION_SPLIT]
    text_prototypes = data["text_prototypes"]

    knn_candidates = []
    for k in CFG.topk_grid:
        validation_scores = full_knn_scores(train_x, train_y, val_x, k)
        knn_candidates.append(
            (accuracy_score(val_y, validation_scores.argmax(1)), -int(k), int(k))
        )
    selected_knn_k = max(knn_candidates)[-1]
    knn_val = full_knn_scores(train_x, train_y, val_x, selected_knn_k)
    knn_evaluation, knn_seconds = timed_scores(
        lambda: full_knn_scores(
            train_x, train_y, evaluation_x, selected_knn_k
        )
    )

    probe_candidates = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        probe = LogisticRegression(
            C=c_value,
            max_iter=2_000,
            solver="lbfgs",
            random_state=CFG.seeds[0],
        )
        probe.fit(train_x, train_y)
        validation_scores = linear_probe_scores(probe, val_x)
        probe_candidates.append(
            (
                accuracy_score(val_y, validation_scores.argmax(1)),
                -float(c_value),
                float(c_value),
                probe,
                validation_scores,
            )
        )
    _, _, selected_probe_c, selected_probe, probe_val = max(
        probe_candidates, key=lambda item: item[:2]
    )
    probe_evaluation, probe_seconds = timed_scores(
        lambda: linear_probe_scores(selected_probe, evaluation_x)
    )

    for seed in CFG.seeds:
        seed_everything(seed)
        memories = {
            "Random memory": fit_or_load_memory(
                train_x,
                train_y,
                CFG.default_budget,
                "random",
                seed,
                encoder_key,
                text_prototypes,
            ),
            "Centroid": fit_or_load_memory(
                train_x,
                train_y,
                1,
                "centroid",
                seed,
                encoder_key,
                text_prototypes,
            ),
            "KMeans medoids": fit_or_load_memory(
                train_x,
                train_y,
                CFG.default_budget,
                "medoid",
                seed,
                encoder_key,
                text_prototypes,
            ),
            "Facility selection (no reliability)": fit_or_load_memory(
                train_x,
                train_y,
                CFG.default_budget,
                "facility_no_reliability",
                seed,
                encoder_key,
                text_prototypes,
            ),
            "EvidenceMem v2 selection": fit_or_load_memory(
                train_x,
                train_y,
                CFG.default_budget,
                "evidencemem",
                seed,
                encoder_key,
                text_prototypes,
            ),
        }

        standard_settings = {}
        for memory_name, memory in memories.items():
            setting = tune_memory(memory, val_x, val_y, text_prototypes)
            standard_settings[memory_name] = setting
            for k in CFG.topk_grid:
                for alpha in CFG.alpha_grid:
                    scores = memory_scores(
                        memory, val_x, alpha, k, text_prototypes
                    )[0]
                    tuning_rows.append(
                        {
                            "encoder_key": encoder_key,
                            "resolution": resolution,
                            "seed": int(seed),
                            "method": memory_name,
                            "alpha": float(alpha),
                            "k": int(k),
                            "validation_accuracy": float(
                                accuracy_score(val_y, scores.argmax(1))
                            ),
                        }
                    )

        v4_setting, primary_memory, v4_tuning = tune_v4_memory(
            train_x,
            train_y,
            CFG.default_budget,
            seed,
            encoder_key,
            val_x,
            val_y,
            text_prototypes,
        )
        for record in v4_tuning:
            reliability_tuning_rows.append(
                {
                    "encoder_key": encoder_key,
                    "resolution": resolution,
                    "seed": int(seed),
                    **record,
                }
            )
        primary_memories[(encoder_key, int(seed))] = primary_memory
        selected_hyperparameters[f"{encoder_key}_s{seed}"] = {
            "standard_methods": standard_settings,
            "evidencemem_v4": v4_setting,
            "full_knn_k": selected_knn_k,
            "linear_probe_c": selected_probe_c,
        }

        score_sets = {
            "CLIP zero-shot": {
                "val": val_x @ text_prototypes.T,
                "evaluation": evaluation_x @ text_prototypes.T,
                "seconds": 0.0,
                "setting": {},
                "memory": None,
                "selected": None,
            },
            "Full kNN": {
                "val": knn_val,
                "evaluation": knn_evaluation,
                "seconds": knn_seconds,
                "setting": {"k": selected_knn_k},
                "memory": None,
                "selected": None,
            },
            "Linear probe": {
                "val": probe_val,
                "evaluation": probe_evaluation,
                "seconds": probe_seconds,
                "setting": {"c": selected_probe_c},
                "memory": None,
                "selected": None,
            },
        }

        for memory_name, memory in memories.items():
            setting = standard_settings[memory_name]
            validation_scores = memory_scores(
                memory,
                val_x,
                setting["alpha"],
                setting["k"],
                text_prototypes,
            )[0]
            evaluation_scores, elapsed = timed_scores(
                lambda memory=memory, setting=setting: memory_scores(
                    memory,
                    evaluation_x,
                    setting["alpha"],
                    setting["k"],
                    text_prototypes,
                )[0]
            )
            selected = nearest_memory_indices(
                memory, evaluation_x, setting["k"]
            )
            score_sets[f"{memory_name} fused"] = {
                "val": validation_scores,
                "evaluation": evaluation_scores,
                "seconds": elapsed,
                "setting": setting,
                "memory": memory,
                "selected": selected,
            }

        global_setting = v4_setting["global_setting"]
        v4_global_val = memory_scores(
            primary_memory,
            val_x,
            global_setting["alpha"],
            global_setting["k"],
            text_prototypes,
        )[0]
        v4_global_evaluation, v4_global_seconds = timed_scores(
            lambda: memory_scores(
                primary_memory,
                evaluation_x,
                global_setting["alpha"],
                global_setting["k"],
                text_prototypes,
            )[0]
        )
        global_selected = nearest_memory_indices(
            primary_memory, evaluation_x, global_setting["k"]
        )
        score_sets["EvidenceMem v4 global fusion"] = {
            "val": v4_global_val,
            "evaluation": v4_global_evaluation,
            "seconds": v4_global_seconds,
            "setting": global_setting,
            "memory": primary_memory,
            "selected": global_selected,
        }

        v4_val = v4_score_variants(
            primary_memory, val_x, v4_setting, text_prototypes
        )
        v4_evaluation, v4_seconds = timed_scores(
            lambda: v4_score_variants(
                primary_memory, evaluation_x, v4_setting, text_prototypes
            )
        )
        for method_name, variant_name in (
            ("EvidenceMem v4 class-conditional visual", "visual"),
            ("EvidenceMem v4 fixed fusion", "fixed"),
            ("EvidenceMem v4 continuous fusion", "continuous"),
        ):
            score_sets[method_name] = {
                "val": v4_val[variant_name]["scores"],
                "evaluation": v4_evaluation[variant_name]["scores"],
                "seconds": v4_seconds,
                "setting": v4_setting,
                "memory": primary_memory,
                # Use global neighbours for a non-tautological evidence metric.
                "selected": global_selected,
                "decision_selected": v4_evaluation[variant_name]["selected"],
                "query_reliability": v4_evaluation[variant_name][
                    "query_reliability"
                ],
                "text_weight": v4_evaluation[variant_name]["text_weight"],
            }

        tip_cache = memory_to_arrays(memories["Random memory"])
        tip_candidates = []
        for beta in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            for cache_weight in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                scores = tip_adapter_scores(
                    tip_cache,
                    val_x,
                    text_prototypes,
                    beta=beta,
                    cache_weight=cache_weight,
                )
                tip_candidates.append(
                    (
                        accuracy_score(val_y, scores.argmax(1)),
                        -beta,
                        -cache_weight,
                        beta,
                        cache_weight,
                        scores,
                    )
                )
        _, _, _, tip_beta, tip_weight, tip_val = max(
            tip_candidates, key=lambda item: item[:3]
        )
        tip_evaluation, tip_seconds = timed_scores(
            lambda: tip_adapter_scores(
                tip_cache,
                evaluation_x,
                text_prototypes,
                beta=tip_beta,
                cache_weight=tip_weight,
            )
        )
        tip_selected = nearest_memory_indices(
            memories["Random memory"],
            evaluation_x,
            min(10, len(tip_cache.labels)),
        )
        score_sets["Tip-Adapter (matched cache)"] = {
            "val": tip_val,
            "evaluation": tip_evaluation,
            "seconds": tip_seconds,
            "setting": {"beta": tip_beta, "cache_weight": tip_weight},
            "memory": memories["Random memory"],
            "selected": tip_selected,
        }

        if set(score_sets) != set(MAIN_METHODS):
            raise AssertionError(
                f"Method registry mismatch: {sorted(score_sets)}"
            )
        for method_name, packet in score_sets.items():
            temperature = select_temperature(packet["val"], val_y)
            confidence_threshold = select_confidence_threshold(
                packet["val"], temperature
            )
            setting = packet["setting"]
            evidence_precision = np.nan
            if packet["memory"] is not None and packet["selected"] is not None:
                evidence_precision = evidence_precision_at_k(
                    packet["memory"], packet["selected"], evaluation_y
                )
            extra = {
                "stored_examples": (
                    len(packet["memory"]["labels"])
                    if packet["memory"] is not None
                    else (len(train_y) if method_name == "Full kNN" else 0)
                ),
                "budget_per_class": (
                    int(packet["memory"]["budget"])
                    if packet["memory"] is not None
                    else np.nan
                ),
                "selected_alpha": setting.get("alpha", np.nan),
                "selected_k": setting.get("k", np.nan),
                "selected_class_k": setting.get("class_k", np.nan),
                "selected_reliability_power": setting.get(
                    "reliability_power", np.nan
                ),
                "global_neighbor_label_precision_at_k": evidence_precision,
                "query_reliability_mean": float(
                    np.mean(packet.get("query_reliability", np.array([np.nan])))
                ),
                "text_weight_mean": float(
                    np.mean(packet.get("text_weight", np.array([np.nan])))
                ),
                "inference_ms_per_query": float(
                    1_000.0 * packet["seconds"] / len(evaluation_y)
                ),
            }
            row, prediction, probabilities = evaluate_scores(
                method_name,
                packet["evaluation"],
                evaluation_y,
                temperature,
                confidence_threshold,
                seed,
                encoder_key,
                resolution,
                extra,
            )
            classification_rows.append(row)
            prediction_bank[(encoder_key, int(seed), method_name)] = prediction
            prediction_path = RUN_DIR / (
                f"predictions_{encoder_key}_s{seed}_{safe_filename(method_name)}.npz"
            )
            np.savez_compressed(
                prediction_path,
                labels=evaluation_y,
                predictions=prediction,
                probabilities=probabilities,
                scores=packet["evaluation"],
            )

        continuous_packet = score_sets["EvidenceMem v4 continuous fusion"]
        continuous_temperature = select_temperature(
            continuous_packet["val"], val_y
        )
        qualitative_bank[(encoder_key, int(seed))] = {
            "prediction": prediction_bank[
                (encoder_key, int(seed), "EvidenceMem v4 continuous fusion")
            ],
            "probabilities": softmax_np(
                continuous_packet["evaluation"], continuous_temperature
            ),
            "query_reliability": continuous_packet["query_reliability"],
            "text_weight": continuous_packet["text_weight"],
            "selected": continuous_packet["decision_selected"],
            "labels": evaluation_y,
        }
        atomic_csv(
            pd.DataFrame(classification_rows),
            RUN_DIR / "classification_results.csv",
        )
        atomic_csv(
            pd.DataFrame(tuning_rows),
            RUN_DIR / "fusion_topk_validation.csv",
        )
        atomic_csv(
            pd.DataFrame(reliability_tuning_rows),
            RUN_DIR / "reliability_tuning.csv",
        )
        atomic_json(
            RUN_DIR / "selected_hyperparameters.json",
            selected_hyperparameters,
        )
        journal(
            "classification_seed_complete",
            encoder_key=encoder_key,
            seed=int(seed),
        )

classification_df = pd.DataFrame(classification_rows)
classification_summary_df = (
    classification_df.groupby(
        ["encoder_key", "resolution", "method"], as_index=False
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        global_neighbor_label_precision_mean=(
            "global_neighbor_label_precision_at_k",
            "mean",
        ),
        aurc_mean=("aurc", "mean"),
        ece_mean=("ece_15", "mean"),
        ece_uncalibrated_mean=("ece_15_uncalibrated", "mean"),
        selective_coverage_mean=("selective_coverage", "mean"),
        selective_accuracy_mean=("selective_accuracy", "mean"),
        inference_ms_per_query_mean=("inference_ms_per_query", "mean"),
    )
    .fillna(0.0)
    .sort_values(["encoder_key", "accuracy_mean"], ascending=[True, False])
)
atomic_csv(classification_summary_df, RUN_DIR / "classification_summary.csv")
calibration_summary_df = classification_df[
    [
        "encoder_key",
        "resolution",
        "seed",
        "method",
        "temperature",
        "nll_uncalibrated",
        "nll",
        "ece_15_uncalibrated",
        "ece_15",
        "confidence_threshold_90pct_validation_coverage",
        "selective_coverage",
        "selective_accuracy",
        "aurc",
    ]
].copy()
atomic_csv(calibration_summary_df, RUN_DIR / "calibration_summary.csv")

development_scores = classification_summary_df[
    classification_summary_df["method"] == "EvidenceMem v4 continuous fusion"
].sort_values(
    ["accuracy_mean", "macro_f1_mean", "ece_mean", "encoder_key"],
    ascending=[False, False, True, True],
)
SELECTED_ENCODER_KEY = str(development_scores.iloc[0]["encoder_key"])
selection_record = {
    "evaluation_stage": EVALUATION_STAGE,
    "evaluation_split": EVALUATION_SPLIT,
    "selected_encoder_key": SELECTED_ENCODER_KEY,
    "selection_method": "EvidenceMem v4 continuous fusion",
    "selection_rule": "highest mean development accuracy; macro-F1, ECE, then key break ties",
    "frozen_for_confirmatory": EVALUATION_STAGE == "confirmatory",
    "warning": (
        "Development selection is exploratory and cannot support a final claim."
        if EVALUATION_STAGE == "development"
        else "This run evaluates the pre-frozen encoder on the untouched confirmatory split."
    ),
}
if EVALUATION_STAGE == "confirmatory":
    if SELECTED_ENCODER_KEY != CONFIRMATORY_ENCODER_KEY:
        raise AssertionError("Confirmatory encoder changed after it was frozen.")
atomic_json(RUN_DIR / "development_selection.json", selection_record)

paired_results = []
for encoder_key, data in ENCODER_DATA.items():
    labels = data[EVALUATION_SPLIT][1]
    for seed in CFG.seeds:
        evidence_correct = (
            prediction_bank[
                (encoder_key, int(seed), "EvidenceMem v4 continuous fusion")
            ]
            == labels
        )
        for baseline in (
            "Facility selection (no reliability) fused",
            "KMeans medoids fused",
            "Tip-Adapter (matched cache)",
            "Linear probe",
            "EvidenceMem v4 global fusion",
            "EvidenceMem v4 fixed fusion",
        ):
            baseline_correct = (
                prediction_bank[(encoder_key, int(seed), baseline)] == labels
            )
            paired_results.append(
                {
                    "comparison_type": "method",
                    "encoder_key": encoder_key,
                    "seed": int(seed),
                    "comparison": (
                        "EvidenceMem v4 continuous fusion vs " + baseline
                    ),
                    "bootstrap": paired_bootstrap(
                        evidence_correct,
                        baseline_correct,
                        seed=CFG.sample_seed + int(seed),
                    ),
                    "mcnemar": mcnemar_exact(
                        evidence_correct, baseline_correct
                    ),
                }
            )

reference_encoder = "clip_b32_224"
if reference_encoder in ENCODER_DATA:
    reference_labels = ENCODER_DATA[reference_encoder][EVALUATION_SPLIT][1]
    for encoder_key, data in ENCODER_DATA.items():
        if encoder_key == reference_encoder:
            continue
        labels = data[EVALUATION_SPLIT][1]
        if not np.array_equal(reference_labels, labels):
            raise AssertionError("Encoder conditions do not share evaluation labels.")
        for seed in CFG.seeds:
            encoder_correct = (
                prediction_bank[
                    (encoder_key, int(seed), "EvidenceMem v4 continuous fusion")
                ]
                == labels
            )
            reference_correct = (
                prediction_bank[
                    (
                        reference_encoder,
                        int(seed),
                        "EvidenceMem v4 continuous fusion",
                    )
                ]
                == labels
            )
            paired_results.append(
                {
                    "comparison_type": "encoder",
                    "encoder_key": encoder_key,
                    "reference_encoder": reference_encoder,
                    "seed": int(seed),
                    "comparison": f"{encoder_key} vs {reference_encoder}",
                    "bootstrap": paired_bootstrap(
                        encoder_correct,
                        reference_correct,
                        seed=CFG.sample_seed + int(seed),
                    ),
                    "mcnemar": mcnemar_exact(
                        encoder_correct, reference_correct
                    ),
                }
            )

adjusted = benjamini_hochberg(
    [row["mcnemar"]["p_exact"] for row in paired_results]
)
for row, adjusted_p in zip(paired_results, adjusted, strict=True):
    row["mcnemar"]["p_bh"] = float(adjusted_p)
atomic_json(RUN_DIR / "paired_tests.json", paired_results)

encoder_deltas = classification_df.pivot_table(
    index=["seed", "method"],
    columns="encoder_key",
    values=["accuracy", "macro_f1", "ece_15", "aurc"],
).reset_index()
encoder_deltas.columns = [
    "_".join(str(part) for part in column if str(part) != "")
    if isinstance(column, tuple)
    else str(column)
    for column in encoder_deltas.columns
]
for encoder_key in ENCODER_DATA:
    if encoder_key == reference_encoder:
        continue
    for metric in ("accuracy", "macro_f1", "ece_15", "aurc"):
        target = f"{metric}_{encoder_key}"
        reference = f"{metric}_{reference_encoder}"
        if target in encoder_deltas and reference in encoder_deltas:
            encoder_deltas[f"{metric}_delta_{encoder_key}_minus_{reference_encoder}"] = (
                encoder_deltas[target] - encoder_deltas[reference]
            )
atomic_csv(encoder_deltas, RUN_DIR / "encoder_deltas.csv")

display(classification_summary_df)
display(pd.DataFrame([selection_record]))
