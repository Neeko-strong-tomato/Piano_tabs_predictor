print("Training example")

import torch
import src.Models.model
import src.Trainings.train as train
import src.metrics.graph as graph

from src.Trainings.train import move_to_device
from src.dataLoader.dataLoader import NSynthDataset
from src.metrics.metrics import instrument_accuracy

from torch.utils.data import DataLoader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():

    # ======================
    # CONFIG
    # ======================
    MODE = "instrument"
    num_epochs = 4
    batch_size = 64
    lr = 1e-3

    # ======================
    # DEVICE
    # ======================
    device = get_device()
    print(f"Using device: {device}")

    # ======================
    # MODEL
    # ======================
    model = src.Models.model.SimpleCNN(
        mode=MODE,
        num_families=10,
        num_sources=3,
        use_batchnorm=True
    ).to(device)

    print(
        "Output size:",
        model.family_head.out_features,
        "families and",
        model.source_head.out_features,
        "sources"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # ======================
    # DATASET
    # ======================
    dataset = NSynthDataset(
        tfrecord_path="nsynth-valid.tfrecord",
        wav_dir="nsynth-valid/audio",
        index_path="nsynth-valid.idx",
        mel_cache_dir="cache/mels_aug0.15_0.01",
        max_samples=12678,
        mode=MODE
    )

    # ======================
    # SPLIT : TRAIN / VAL / TEST
    # ======================
    N = len(dataset)
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)
    test_size = N - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    TrainDataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    ValDataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    TestDataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ======================
    # TRAINING
    # ======================
    trained_model, history = train.train(
        model,
        TrainDataLoader,
        ValDataLoader,
        criterion,
        optimizer,
        num_epochs,
        metrics=[instrument_accuracy],
        device=device,
        mode=MODE
    )

    graph.plot_training_metrics(history)

    # ======================
    # TEST EVALUATION
    # ======================
    all_preds_family = []
    all_true_family = []

    all_preds_source = []
    all_true_source = []

    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in TestDataLoader:
            test_inputs = move_to_device(test_inputs, device)
            test_labels = move_to_device(test_labels, device)

            outputs = model(test_inputs)

            family_pred = outputs["family"].argmax(dim=1).cpu().numpy()
            source_pred = outputs["source"].argmax(dim=1).cpu().numpy()

            family_true = test_labels["family"].cpu().numpy()
            source_true = test_labels["source"].cpu().numpy()

            all_preds_family.extend(family_pred)
            all_true_family.extend(family_true)

            all_preds_source.extend(source_pred)
            all_true_source.extend(source_true)

    # ======================
    # CONFUSION MATRICES (TEST ONLY)
    # ======================
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        # FAMILY
        cm_family = confusion_matrix(all_true_family, all_preds_family)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_family, cmap="Blues")
        plt.title("Confusion Matrix – Instrument Family (TEST)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        # SOURCE
        cm_source = confusion_matrix(all_true_source, all_preds_source)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_source, cmap="Blues")
        plt.title("Confusion Matrix – Instrument Source (TEST)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("[INFO] matplotlib / sklearn non installés.")

    print("Finished.")


if __name__ == "__main__":
    main()
