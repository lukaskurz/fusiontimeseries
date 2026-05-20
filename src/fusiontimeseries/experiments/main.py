# run with uv run src/fusiontimeseries/experiments/main.py

from fusiontimeseries.loralib.layers import RSSBilinearLoRA
import torch
from torch import nn
from fusiontimeseries.experiments.config import FinetuningConfig
from fusiontimeseries.experiments.config import get_output_dir
from fusiontimeseries.experiments.dataset import FluxDataset
from fusiontimeseries.experiments.model import get_model

from fusiontimeseries.experiments.model import evaluate
from fusiontimeseries.experiments.vizualization import plot_forecast
import json
from fusiontimeseries.loralib.utils import lora_state_dict

from fusiontimeseries.experiments.trainer import TimesFMTrainer
from datetime import datetime

run_configs = [
    ############################################## no subsampling ##############################################
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 11,
    #     "train_context_cutoffs": [11, 139, 267, 395, 523, 651, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 21,
    #     "train_context_cutoffs": [21, 149, 277, 405, 533, 661, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 41,
    #     "train_context_cutoffs": [41, 169, 297, 425, 553, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 80,
    #     "train_context_cutoffs": [80, 208, 336, 464, 592, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 11,
    #     "train_context_cutoffs": [11, 139, 267, 395, 523, 651, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 21,
    #     "train_context_cutoffs": [21, 149, 277, 405, 533, 661, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 41,
    #     "train_context_cutoffs": [41, 169, 297, 425, 553, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 80,
    #     "train_context_cutoffs": [80, 208, 336, 464, 592, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": Linear,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 11,
    #     "train_context_cutoffs": [11, 139, 267, 395, 523, 651, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": Linear,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 21,
    #     "train_context_cutoffs": [21, 149, 277, 405, 533, 661, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": Linear,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 41,
    #     "train_context_cutoffs": [41, 169, 297, 425, 553, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": Linear,
    # },
    # {
    #     "run_folder_prefix": "FullContext-",
    #     "eval_context_cutoff": 80,
    #     "train_context_cutoffs": [80, 208, 336, 464, 592, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": Linear,
    # },
    ############################################## subsampling ##############################################
    # {
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 70, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 70, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": Linear,
    # },
    # {
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 70, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 11,
    #     "train_context_cutoffs": [11, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 11,
    #     "train_context_cutoffs": [11, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": Linear,
    # },
    # {
    #     "eval_context_cutoff": 11,
    #     "train_context_cutoffs": [11, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 21,
    #     "train_context_cutoffs": [21, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 21,
    #     "train_context_cutoffs": [21, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": Linear,
    # },
    # {
    #     "eval_context_cutoff": 21,
    #     "train_context_cutoffs": [21, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 41,
    #     "train_context_cutoffs": [41, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 41,
    #     "train_context_cutoffs": [41, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": Linear,
    # },
    # {
    #     "eval_context_cutoff": 41,
    #     "train_context_cutoffs": [41, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": RSSBilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 80,
    #     "train_context_cutoffs": [80, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": BilinearLoRA,
    # },
    # {
    #     "eval_context_cutoff": 80,
    #     "train_context_cutoffs": [80, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": Linear,
    # },
    # {
    #     "eval_context_cutoff": 80,
    #     "train_context_cutoffs": [80, 139],
    #     "subsampling": True,
    #     "context_length": 288,
    #     "adapter": RSSBilinearLoRA,
    # },
    ############################################## data scaling ##############################################
    # {
    #     "run_folder_prefix": "DataScaling-b1-",
    #     "eval_context_cutoff": 1,
    #     "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
    #     "subsampling": False,
    #     "context_length": 800,
    #     "adapter": RSSBilinearLoRA,
    #     "train_namespaces": [
    #         "gyroswin_train",
    #         "batch_1",
    #     ],
    #     "val_namespaces": ["batch_6"],
    #     "test_namespaces": ["batch_9"],
    # },
    {
        "run_folder_prefix": "DataScaling-b12-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
    {
        "run_folder_prefix": "DataScaling-b123-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
            "batch_3",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
    {
        "run_folder_prefix": "DataScaling-b1234-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
    {
        "run_folder_prefix": "DataScaling-b12345-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
            "batch_5",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
    {
        "run_folder_prefix": "DataScaling-b123457-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
            "batch_5",
            "batch_7",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
    {
        "run_folder_prefix": "DataScaling-b1234578-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
            "batch_5",
            "batch_7",
            "batch_8",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
    {
        "run_folder_prefix": "DataScaling-b123457810-",
        "eval_context_cutoff": 1,
        "train_context_cutoffs": [1, 129, 257, 385, 513, 641, 672],
        "subsampling": False,
        "context_length": 800,
        "adapter": RSSBilinearLoRA,
        "train_namespaces": [
            "gyroswin_train",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
            "batch_5",
            "batch_7",
            "batch_8",
            "batch_10",
        ],
        "val_namespaces": ["batch_6"],
        "test_namespaces": ["batch_9"],
    },
]


if __name__ == "__main__":
    for run in run_configs:
        RUN_FOLDER_PREFIX = run.get("run_folder_prefix", "")
        EVAL_CONTEXT_CUTOFF = run["eval_context_cutoff"]
        TRAIN_CONTEXT_CUTOFFS = run["train_context_cutoffs"]
        SUBSAMPLING = run["subsampling"]
        CONTEXT_LENGTH = run["context_length"]
        Adapter = run["adapter"]
        TRAIN_NAMESPACE_OVERRIDES = run.get("train_namespaces")
        VAL_NAMESPACE_OVERRIDES = run.get("val_namespaces")
        TEST_NAMESPACES = run.get("test_namespaces")

        print(
            f"Running with eval_context_cutoff={EVAL_CONTEXT_CUTOFF}, subsampling={SUBSAMPLING}, context_length={CONTEXT_LENGTH}, adapter={Adapter.__name__}"
        )
        torch.cuda.empty_cache()

        config = FinetuningConfig(
            eval_context_cutoff=EVAL_CONTEXT_CUTOFF,
            train_context_cutoffs=TRAIN_CONTEXT_CUTOFFS,
            subsampling=SUBSAMPLING,
            context_length=CONTEXT_LENGTH,
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = get_output_dir(
            "./outputs",
            base_folder=f"{RUN_FOLDER_PREFIX}{Adapter.__name__}-{EVAL_CONTEXT_CUTOFF}-{timestamp}",
        )
        print(f"Output directory: {output_dir}")

        print("Loading datasets...")
        if TRAIN_NAMESPACE_OVERRIDES is not None:
            train_dataset = FluxDataset(
                namespaces=TRAIN_NAMESPACE_OVERRIDES, config=config
            )
            print(f"Using train namespaces: {TRAIN_NAMESPACE_OVERRIDES}")
            print(f"Samples in train dataset: {len(train_dataset)}")
        else:
            train_dataset = FluxDataset(namespaces=["gyroswin_train"], config=config)
            print("Using Gyroswin train namespace")
            print(f"Samples in train dataset: {len(train_dataset)}")

        if VAL_NAMESPACE_OVERRIDES is not None:
            print(f"Using val namespaces: {VAL_NAMESPACE_OVERRIDES}")
            val_dataset = FluxDataset(namespaces=VAL_NAMESPACE_OVERRIDES, config=config)
        else:
            print("Using Gyroswin val namespace")
            val_dataset = FluxDataset(namespaces=["gyroswin_val"], config=config)

        id_test_dataset = FluxDataset(namespaces=["gyroswin_id"], config=config)
        ood_test_dataset = FluxDataset(namespaces=["gyroswin_ood"], config=config)
        if TEST_NAMESPACES is not None:
            print(f"Using test namespaces: {TEST_NAMESPACES}")
            test_dataset = FluxDataset(namespaces=TEST_NAMESPACES, config=config)

        print("Initializing model...")
        model: nn.Module = get_model(
            config=config,
            output_dir=output_dir,
            device="cuda",
            Adapter=Adapter,
        )

        trainer = TimesFMTrainer(
            model=model,  # type: ignore
            train_args=config.get_training_arguments(
                output_dir=output_dir, load_best_model_at_end=False
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            config=config,
        )
        with open(output_dir / "training_args.json", "w") as f:
            json.dump(trainer.args.to_dict(), f, indent=4)
        config.save_config(output_dir / "fts_config.json")

        train_output = trainer.train()

        with open(output_dir / "train_summary.json", "w") as f:
            json.dump(train_output._asdict(), f, indent=4)

        lora_weights = lora_state_dict(model)
        torch.save(lora_weights, output_dir / "lora_weights.pt")

        print("Evaluating model...")
        trained_model = trainer.model.eval()
        id_results = evaluate(
            model=trained_model,
            config=config,
            data=id_test_dataset.flux_data,
            device="cuda",
        )
        with open(output_dir / "id_test_results.json", "w") as f:
            json.dump(id_results, f, indent=4)

        ood_results = evaluate(
            model=trained_model,
            config=config,
            data=ood_test_dataset.flux_data,
            device="cuda",
        )
        with open(output_dir / "ood_test_results.json", "w") as f:
            json.dump(ood_results, f, indent=4)

        val_results = evaluate(
            model=trained_model,
            config=config,
            data=val_dataset.flux_data,
            device="cuda",
        )
        with open(output_dir / "val_results.json", "w") as f:
            json.dump(val_results, f, indent=4)

        if TEST_NAMESPACES is not None:
            test_results = evaluate(
                model=trained_model,
                config=config,
                data=test_dataset.flux_data,
                device="cuda",
            )
            with open(output_dir / "batch9_test_results.json", "w") as f:
                json.dump(test_results, f, indent=4)

        train_results = evaluate(
            model=trained_model,
            config=config,
            data=train_dataset.flux_data,
            device="cuda",
        )
        with open(output_dir / "train_results.json", "w") as f:
            json.dump(train_results, f, indent=4)

        print("Generating plots...")
        plot_forecast(
            results=id_results,
            simulations=[
                "3000",  # 8
                "3001",  # 115
            ],
            config=config,
            output_dir=output_dir,
            show_plots=False,
        )
        plot_forecast(
            results=val_results,
            simulations=[
                "2001",  # 100
                "2002",  # 200
            ],
            config=config,
            output_dir=output_dir,
            show_plots=False,
        )
        plot_forecast(
            results=ood_results,
            simulations=[
                "4001",  # 1
                "4003",  # 3
            ],
            config=config,
            output_dir=output_dir,
            show_plots=False,
        )

    torch.cuda.empty_cache()
