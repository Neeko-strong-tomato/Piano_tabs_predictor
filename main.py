print("Training example")


import src.Models.model
import src.Trainings.train as train
import src.metrics.metrics as metrics
import src.metrics.graph as graph
import torch

from src.dataLoader.dataLoader import NSynthDataset
from torch.utils.data import DataLoader

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():

    device = get_device()
    print(f"Using device: {device}")
    model = src.Models.model.SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 12
    batch_size = 64
    data_dir = "./src/dataLoader/"
    dataset = NSynthDataset(
        tfrecord_path="nsynth-valid.tfrecord",
        wav_dir="nsynth-valid/audio",
        index_path="nsynth-valid.idx",
        max_samples=12000
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
            metrics=[metrics.mae, metrics.mse, metrics.accuracy],
            device=device
        )
    graph.plot_training_metrics(history)

    # Analyse avancée : collecte des prédictions et labels sur la validation
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for val_inputs, val_labels in ValDataLoader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            pred_class = val_outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred_class)
            all_labels.extend(val_labels.cpu().numpy())

    # Matrice de confusion et histogramme des erreurs
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import numpy as np
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, aspect='auto', cmap='Blues')
        plt.title('Matrice de confusion (validation)')
        plt.xlabel('Prédit')
        plt.ylabel('Vrai')
        plt.colorbar()
        plt.show()

        errors = np.array(all_preds) - np.array(all_labels)
        plt.figure()
        plt.hist(errors, bins=30)
        plt.title('Histogramme des erreurs (prédit - vrai)')
        plt.xlabel('Erreur (demi-tons)')
        plt.ylabel('Nombre de samples')
        plt.show()
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