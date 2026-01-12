print("Training example")


import src.Models.model
import src.Trainings.train as train
import src.metrics.metrics as metrics
import src.metrics.graph as graph
import torch
from src.Trainings.train import move_to_device


from src.dataLoader.dataLoader import NSynthDataset
from torch.utils.data import DataLoader
from src.metrics.metrics import instrument_accuracy

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():

    device = get_device()
    MODE = "instrument"
    print(f"Using device: {device}")
    model = src.Models.model.SimpleCNN(
                mode=MODE,
                num_families=10,
                num_sources=3,
                use_batchnorm=True
            ).to(device)

    if MODE == "notes":
        print("Output size (num classes):", model.final_conv.out_channels)
    else:
        print("Output size (num classes):", model.family_head.out_features, "families and", model.source_head.out_features, "sources")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 4
    batch_size = 64
    
    data_dir = "./src/dataLoader/"
    dataset = NSynthDataset(
        tfrecord_path="nsynth-valid.tfrecord",
        wav_dir="nsynth-valid/audio",
        index_path="nsynth-valid.idx",
        mel_cache_dir="cache/mels_aug0.15_0.01",
        max_samples=12678,
        mode=MODE
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    TrainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    ValDataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
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

    # Analyse avancée : collecte des prédictions et labels sur la validation
    all_preds_family = []
    all_true_family = []

    all_preds_source = []
    all_true_source = []

    model.eval()
    with torch.no_grad():
        for val_inputs, val_labels in ValDataLoader:
            val_inputs = move_to_device(val_inputs, device)
            val_labels = move_to_device(val_labels, device)
            val_outputs = model(val_inputs)
            family_pred = val_outputs["family"].argmax(dim=1).cpu().numpy()
            source_pred = val_outputs["source"].argmax(dim=1).cpu().numpy()

            family_true = val_labels["family"].cpu().numpy()
            source_true = val_labels["source"].cpu().numpy()

            all_preds_family.extend(family_pred)
            all_true_family.extend(family_true)

            all_preds_source.extend(source_pred)
            all_true_source.extend(source_true)


    # Matrice de confusion et histogramme des erreurs
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import numpy as np

        # === FAMILY CONFUSION MATRIX ===
        cm_family = confusion_matrix(all_true_family, all_preds_family)
        plt.figure(figsize=(6,5))
        plt.imshow(cm_family, cmap="Blues")
        plt.title("Confusion Matrix – Instrument Family")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        # === SOURCE CONFUSION MATRIX ===
        cm_source = confusion_matrix(all_true_source, all_preds_source)
        plt.figure(figsize=(6,5))
        plt.imshow(cm_source, cmap="Blues")
        plt.title("Confusion Matrix – Instrument Source")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


        #errors = np.array(all_preds) - np.array(all_labels)
        #plt.figure()
        #plt.hist(errors, bins=30)
        #plt.title('Histogramme des erreurs (prédit - vrai)')
        #plt.xlabel('Erreur (demi-tons)')
        #plt.ylabel('Nombre de samples')
        #plt.show()
    except ImportError:
        print("[INFO] matplotlib ou scikit-learn non installés : pas de visualisation avancée.")

    #evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in ValDataLoader:
            val_outputs = model(val_inputs)
            v_loss = criterion(val_outputs, val_labels)
            val_loss += v_loss.item() * val_inputs.size(0)
    val_loss /= len(ValDataLoader.dataset)
    print(f'Final Validation Loss: {val_loss:.4f}')

if __name__ == "__main__":
    main()