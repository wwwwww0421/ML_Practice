

def promote_if_better_model(client, model_name: str, new_version: str, new_score: float, champion_score: float | None) -> bool:
    if champion_score is None or new_score > champion_score:
        client.set_registered_model_alias(model_name, "champion", new_version)
        print(f"PROMOTE! {new_version}: {new_score}!")
        return True
    else:
        return False