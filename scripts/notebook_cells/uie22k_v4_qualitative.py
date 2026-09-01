# ruff: noqa: E501, F821 - executed inside the generated notebook namespace

qualitative_rows = []
encoder_key = SELECTED_ENCODER_KEY
seed = int(CFG.seeds[0])
packet = qualitative_bank[(encoder_key, seed)]
memory = primary_memories[(encoder_key, seed)]
labels = packet["labels"]
prediction = packet["prediction"]
confidence = packet["probabilities"].max(1)
correct = prediction == labels
wrong_indices = np.flatnonzero(~correct)
correct_indices = np.flatnonzero(correct)
selected_examples = []
if len(wrong_indices):
    selected_examples.extend(
        wrong_indices[np.argsort(-confidence[wrong_indices])[:3]].tolist()
    )
if len(correct_indices):
    selected_examples.extend(
        correct_indices[np.argsort(confidence[correct_indices])[:3]].tolist()
    )
if not selected_examples:
    selected_examples = np.argsort(confidence)[:6].tolist()

figure, axes = plt.subplots(
    len(selected_examples), 4, figsize=(10, 2.6 * len(selected_examples))
)
axes = np.atleast_2d(axes)
train_frame = SPLIT_FRAMES["train"]
evaluation_frame = SPLIT_FRAMES[EVALUATION_SPLIT]
for row_number, evaluation_index in enumerate(selected_examples):
    query_row = evaluation_frame.iloc[int(evaluation_index)]
    query_path = DATASET_ROOT / query_row["relative_path"]
    axes[row_number, 0].imshow(Image.open(query_path).convert("RGB"))
    axes[row_number, 0].set_title(
        f"Query: {CLASS_NAMES[int(labels[evaluation_index])]}\n"
        f"Pred: {CLASS_NAMES[int(prediction[evaluation_index])]} "
        f"({confidence[evaluation_index]:.2f})",
        fontsize=8,
    )
    selected_prototypes = packet["selected"][int(evaluation_index), :3]
    selected_prototypes = selected_prototypes[selected_prototypes >= 0]
    source_indices = np.asarray(memory["source_idx"], dtype=int)[
        selected_prototypes
    ]
    evidence_labels = np.asarray(memory["labels"], dtype=int)[
        selected_prototypes
    ]
    record = {
        "encoder_key": encoder_key,
        "resolution": int(ENCODER_DATA[encoder_key]["spec"]["resolution"]),
        "evaluation_stage": EVALUATION_STAGE,
        "seed": seed,
        "evaluation_index": int(evaluation_index),
        "query_path": query_row["relative_path"],
        "true_label": CLASS_NAMES[int(labels[evaluation_index])],
        "predicted_label": CLASS_NAMES[int(prediction[evaluation_index])],
        "correct": bool(correct[evaluation_index]),
        "confidence": float(confidence[evaluation_index]),
        "query_reliability": float(
            packet["query_reliability"][evaluation_index]
        ),
        "text_weight": float(packet["text_weight"][evaluation_index]),
        "evidence_scope": "top prototypes within the predicted class",
    }
    for evidence_rank, (source_index, evidence_label) in enumerate(
        zip(source_indices, evidence_labels, strict=True), start=1
    ):
        evidence_row = train_frame.iloc[int(source_index)]
        evidence_path = DATASET_ROOT / evidence_row["relative_path"]
        axes[row_number, evidence_rank].imshow(
            Image.open(evidence_path).convert("RGB")
        )
        axes[row_number, evidence_rank].set_title(
            f"Decision evidence {evidence_rank}: "
            f"{CLASS_NAMES[int(evidence_label)]}",
            fontsize=8,
        )
        record[f"evidence_{evidence_rank}_path"] = evidence_row[
            "relative_path"
        ]
        record[f"evidence_{evidence_rank}_label"] = CLASS_NAMES[
            int(evidence_label)
        ]
    qualitative_rows.append(record)
for axis in axes.ravel():
    axis.axis("off")
figure.tight_layout()
figure.savefig(
    RUN_DIR / f"qualitative_evidence_{encoder_key}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
qualitative_df = pd.DataFrame(qualitative_rows)
atomic_csv(qualitative_df, RUN_DIR / "qualitative_evidence.csv")
