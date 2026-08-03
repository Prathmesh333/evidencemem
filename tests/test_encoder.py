from evidencemem import canonical_open_clip_model_name


def test_openai_checkpoint_resolves_to_quickgelu_definition() -> None:
    assert canonical_open_clip_model_name("ViT-B-32", "openai") == "ViT-B-32-quickgelu"
    assert (
        canonical_open_clip_model_name("ViT-B-32-quickgelu", "openai")
        == "ViT-B-32-quickgelu"
    )
    assert canonical_open_clip_model_name("ViT-B-32", "laion2b_s34b_b79k") == "ViT-B-32"
