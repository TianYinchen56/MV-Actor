def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {}

    if dataset_name in {
        "Peract2_3dfront_3dwrist",
        "Peract2_3dfront",
        "Peract",
        "PeractTwoCam",
        "HiveformerRLBench",
    }:
        from .rlbench import (
            Peract2Dataset,
            Peract2SingleCamDataset,
            PeractDataset,
            PeractTwoCamDataset,
            HiveformerDataset,
        )

        dataset_classes.update({
            "Peract2_3dfront_3dwrist": Peract2Dataset,
            "Peract2_3dfront": Peract2SingleCamDataset,
            "Peract": PeractDataset,
            "PeractTwoCam": PeractTwoCamDataset,
            "HiveformerRLBench": HiveformerDataset,
        })

    if dataset_name == "RobotWin2":
        from .robotwin import RobotWinDataset

        dataset_classes["RobotWin2"] = RobotWinDataset
    if dataset_name == "RobotWin3DFA":
        from .robotwin import RobotWin3DFADataset

        dataset_classes["RobotWin3DFA"] = RobotWin3DFADataset

    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_classes[dataset_name]
