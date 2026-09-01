# ruff: noqa: E501, F821 - executed inside the generated notebook namespace

budget_rows = []
data = ENCODER_DATA[SELECTED_ENCODER_KEY]
resolution = int(data["spec"]["resolution"])
train_x, train_y = data["train"]
val_x, val_y = data["val"]
evaluation_x, evaluation_y = data[EVALUATION_SPLIT]
text_prototypes = data["text_prototypes"]

for seed in CFG.seeds:
    for budget in CFG.budgets:
        random_memory = fit_or_load_memory(
            train_x,
            train_y,
            budget,
            "random",
            seed,
            SELECTED_ENCODER_KEY,
            text_prototypes,
        )
        facility_memory = fit_or_load_memory(
            train_x,
            train_y,
            budget,
            "facility_no_reliability",
            seed,
            SELECTED_ENCODER_KEY,
            text_prototypes,
        )
        v4_setting, v4_memory, _ = tune_v4_memory(
            train_x,
            train_y,
            budget,
            seed,
            SELECTED_ENCODER_KEY,
            val_x,
            val_y,
            text_prototypes,
        )

        method_packets = []
        for method_name, memory in (
            ("Random memory", random_memory),
            ("Facility selection (no reliability)", facility_memory),
        ):
            setting = tune_memory(memory, val_x, val_y, text_prototypes)
            scores = memory_scores(
                memory,
                evaluation_x,
                setting["alpha"],
                setting["k"],
                text_prototypes,
            )[0]
            selected = nearest_memory_indices(
                memory, evaluation_x, setting["k"]
            )
            method_packets.append(
                (method_name, memory, setting, scores, selected)
            )

        v4_packet = v4_score_variants(
            v4_memory, evaluation_x, v4_setting, text_prototypes
        )["continuous"]
        v4_global_selected = nearest_memory_indices(
            v4_memory,
            evaluation_x,
            v4_setting["global_setting"]["k"],
        )
        method_packets.append(
            (
                "EvidenceMem v4 continuous fusion",
                v4_memory,
                v4_setting,
                v4_packet["scores"],
                v4_global_selected,
            )
        )

        expected_stored = int(budget) * N_CLASSES
        for method_name, memory, setting, scores, selected in method_packets:
            stored_examples = int(len(memory["labels"]))
            if stored_examples != expected_stored:
                raise AssertionError(
                    f"Unmatched budget for {method_name}: "
                    f"{stored_examples} != {expected_stored}"
                )
            prediction = scores.argmax(1)
            budget_rows.append(
                {
                    "encoder_key": SELECTED_ENCODER_KEY,
                    "resolution": resolution,
                    "evaluation_stage": EVALUATION_STAGE,
                    "seed": int(seed),
                    "method": method_name,
                    "budget_per_class": int(budget),
                    "stored_examples": stored_examples,
                    "accuracy": float(
                        accuracy_score(evaluation_y, prediction)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(evaluation_y, prediction)
                    ),
                    "macro_f1": float(
                        f1_score(evaluation_y, prediction, average="macro")
                    ),
                    "global_neighbor_label_precision_at_k": (
                        evidence_precision_at_k(
                            memory, selected, evaluation_y
                        )
                    ),
                    "selected_setting": json.dumps(setting, sort_keys=True),
                }
            )
        atomic_csv(
            pd.DataFrame(budget_rows),
            RUN_DIR / "memory_budget_results.csv",
        )

budget_df = pd.DataFrame(budget_rows)
budget_summary_df = (
    budget_df.groupby(
        ["encoder_key", "resolution", "method", "budget_per_class"],
        as_index=False,
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        global_neighbor_label_precision_mean=(
            "global_neighbor_label_precision_at_k",
            "mean",
        ),
    )
    .fillna(0.0)
)
atomic_csv(budget_summary_df, RUN_DIR / "memory_budget_summary.csv")
display(budget_summary_df)
