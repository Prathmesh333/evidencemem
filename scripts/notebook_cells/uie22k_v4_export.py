# ruff: noqa: E501, F821 - executed inside the generated notebook namespace

selected_methods = [
    "CLIP zero-shot",
    "Full kNN",
    "Linear probe",
    "Facility selection (no reliability) fused",
    "Tip-Adapter (matched cache)",
    "EvidenceMem v4 global fusion",
    "EvidenceMem v4 fixed fusion",
    "EvidenceMem v4 continuous fusion",
]
plot_frame = classification_summary_df[
    classification_summary_df["method"].isin(selected_methods)
].copy()

figure, axis = plt.subplots(figsize=(12, 5.5))
sns.barplot(
    data=plot_frame,
    x="method",
    y="accuracy_mean",
    hue="encoder_key",
    ax=axis,
)
axis.set_xlabel("")
axis.set_ylabel("Top-1 accuracy")
axis.tick_params(axis="x", rotation=28)
axis.legend(title="Native encoder", fontsize=8)
figure.tight_layout()
figure.savefig(RUN_DIR / "main_accuracy.pdf", bbox_inches="tight")
plt.show()

figure, axis = plt.subplots(figsize=(8, 4.5))
for method, group in budget_summary_df.groupby("method"):
    group = group.sort_values("budget_per_class")
    axis.plot(
        group["budget_per_class"],
        group["accuracy_mean"],
        marker="o",
        label=method,
    )
axis.set_xlabel("Stored images per class")
axis.set_ylabel("Top-1 accuracy")
axis.set_title(f"Matched memory budget: {SELECTED_ENCODER_KEY}")
axis.legend(fontsize=8)
figure.tight_layout()
figure.savefig(RUN_DIR / "memory_budget_accuracy.pdf", bbox_inches="tight")
plt.show()

calibration_plot = (
    calibration_summary_df.groupby(["encoder_key", "method"], as_index=False)
    .agg(
        ece_before=("ece_15_uncalibrated", "mean"),
        ece_after=("ece_15", "mean"),
        nll_before=("nll_uncalibrated", "mean"),
        nll_after=("nll", "mean"),
    )
)
calibration_plot = calibration_plot[
    calibration_plot["method"].isin(selected_methods)
]
calibration_long = calibration_plot.melt(
    id_vars=["encoder_key", "method"],
    value_vars=["ece_before", "ece_after"],
    var_name="calibration",
    value_name="ece_15",
)
figure, axes = plt.subplots(
    1,
    len(ENCODER_DATA),
    figsize=(max(7, 5 * len(ENCODER_DATA)), 4.5),
    squeeze=False,
)
for axis, (encoder_key, group) in zip(
    axes.ravel(), calibration_long.groupby("encoder_key"), strict=False
):
    sns.barplot(
        data=group,
        x="method",
        y="ece_15",
        hue="calibration",
        ax=axis,
    )
    axis.set_title(encoder_key)
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=55, labelsize=7)
    axis.legend(title="", fontsize=7)
figure.tight_layout()
figure.savefig(RUN_DIR / "calibration_ece.pdf", bbox_inches="tight")
plt.show()

v4_summary = classification_summary_df[
    classification_summary_df["method"]
    == "EvidenceMem v4 continuous fusion"
].set_index("encoder_key")
result_summary = {
    "evaluation_stage": EVALUATION_STAGE,
    "evaluation_split": EVALUATION_SPLIT,
    "selected_encoder_key": SELECTED_ENCODER_KEY,
    "accuracy_by_encoder": {
        str(index): float(row["accuracy_mean"])
        for index, row in v4_summary.iterrows()
    },
    "macro_f1_by_encoder": {
        str(index): float(row["macro_f1_mean"])
        for index, row in v4_summary.iterrows()
    },
    "ece_by_encoder": {
        str(index): float(row["ece_mean"])
        for index, row in v4_summary.iterrows()
    },
}

expected_classification_rows = (
    len(ENCODER_DATA) * len(CFG.seeds) * len(MAIN_METHODS)
)
expected_split_counts = {
    "train": CFG.train_per_class,
    "val": CFG.val_per_class,
    "development": CFG.development_per_class,
    "confirmatory": CFG.confirmatory_per_class,
}
observed_split_counts = (
    manifest_df.groupby(["label", "split"]).size().unstack(fill_value=0)
)
manifest_balanced = (
    set(observed_split_counts.index) == set(CLASS_NAMES)
    and set(observed_split_counts.columns) == set(expected_split_counts)
    and all(
        int(observed_split_counts.at[class_name, split_name])
        == expected_count
        for class_name in CLASS_NAMES
        for split_name, expected_count in expected_split_counts.items()
    )
)
integrity_checks = {
    "manifest_balanced": bool(manifest_balanced),
    "manifest_has_no_exact_duplicates": not bool(
        manifest_df["sha256"].duplicated().any()
    ),
    "classification_rows_complete": len(classification_df)
    == expected_classification_rows,
    "method_registry_complete": set(classification_df["method"])
    == set(MAIN_METHODS),
    "all_seeds_complete": classification_df["seed"].nunique()
    == len(CFG.seeds),
    "all_encoders_complete": set(classification_df["encoder_key"])
    == set(ENCODER_DATA),
    "paired_tests_present": len(paired_results) > 0,
    "budget_rows_complete": len(budget_df)
    == len(CFG.seeds) * len(CFG.budgets) * 3,
    "calibration_rows_complete": len(calibration_summary_df)
    == expected_classification_rows,
}
run_complete = all(integrity_checks.values())
claim_gate = {
    "protocol_id": PROTOCOL_ID,
    "protocol_revision": PROTOCOL_REVISION,
    "manifest_id": MANIFEST_ID,
    "mode": CFG.mode,
    "evaluation_stage": EVALUATION_STAGE,
    "integrity_checks": integrity_checks,
    "run_complete": run_complete,
    "ready_for_development_selection": run_complete
    and CFG.mode == "paper"
    and EVALUATION_STAGE == "development",
    "ready_for_final_claims": run_complete
    and CFG.mode == "paper"
    and EVALUATION_STAGE == "confirmatory",
    "result_summary": result_summary,
    "warning": (
        "Development results may select one encoder but are not final evidence. "
        "Freeze the selected key before the confirmatory run."
        if EVALUATION_STAGE == "development"
        else "Confirmatory results are valid only if no tuning follows this run."
    ),
}
atomic_json(RUN_DIR / "claim_gate.json", claim_gate)
if not run_complete:
    failed = [name for name, passed in integrity_checks.items() if not passed]
    raise RuntimeError(f"Run integrity gate failed: {failed}")

# Record the completion event before finalization so the archived journal is the
# same byte sequence covered by the run manifest.  Earlier releases appended
# this event after the zip was built, which left the loose and archived journals
# one line apart.
archive_path = RUN_DIR.with_suffix(".zip")
journal("run_complete", archive=str(archive_path), results=result_summary)

# Hash every top-level result, including all per-example prediction arrays and
# qualitative figures.  The original v4 release used a hand-maintained list and
# therefore omitted those files even though the archive itself was intact.
required_artifacts = sorted(
    path.name
    for path in RUN_DIR.iterdir()
    if path.is_file() and path.name != "run_manifest.json"
)
completed_manifest = finalize_run_manifest(
    RUN_MANIFEST,
    run_directory=RUN_DIR,
    required_artifacts=required_artifacts,
)
atomic_write_json(RUN_DIR / "run_manifest.json", completed_manifest)
archive_path = Path(shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR))
print(json.dumps(claim_gate, indent=2))
print("Complete artifact directory:", RUN_DIR)
print("Downloadable archive:", archive_path)
try:
    from IPython.display import FileLink, display

    display(FileLink(str(archive_path)))
except Exception:
    pass
