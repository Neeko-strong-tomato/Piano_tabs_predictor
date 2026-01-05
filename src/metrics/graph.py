import matplotlib.pyplot as plt

def plot_training_metrics(history):
    """
    Plots training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')

    if 'metrics' in history:
        for metric_name, values in history['metrics'].items():
            plt.plot(values, label=f'Val {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()