import torch
from tqdm import trange

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def train(
        model,
        TrainDataLoader,
        ValDataLoader,
        criterion,
        optimizer,
        num_epochs,
        metrics=None,
        device=None,
        mode="notes"
    ):

    model.train()
    history = {'train_loss': [], 'val_loss': [], 'metrics': {metric.__name__: [] for metric in metrics} if metrics else {}}

    for epoch in trange(num_epochs, desc="Epochs", unit="epoch"):

        epoch_loss = 0.0
        for inputs, labels in TrainDataLoader:
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if mode == "notes":
                loss = criterion(outputs, labels)

            elif mode == "instrument":
                labels["family"] = labels["family"].view(-1)
                labels["source"] = labels["source"].view(-1)

                loss = (
                    criterion(outputs["family"], labels["family"])
                    + criterion(outputs["source"], labels["source"])
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(TrainDataLoader.dataset)
        history['train_loss'].append(epoch_loss)


        # Validation phase
        if ValDataLoader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_inputs, val_labels in ValDataLoader:
                    val_inputs = move_to_device(val_inputs, device)
                    val_labels = move_to_device(val_labels, device)

                    val_outputs = model(val_inputs)

                    if mode == "notes":
                        v_loss = criterion(val_outputs, val_labels)

                    elif mode == "instrument":
                        val_labels["family"] = val_labels["family"].view(-1)
                        val_labels["source"] = val_labels["source"].view(-1)

                        v_loss = (
                            criterion(val_outputs["family"], val_labels["family"])
                            + criterion(val_outputs["source"], val_labels["source"])
                        )

                    val_loss += v_loss.item() * val_inputs.size(0)

            val_loss /= len(ValDataLoader.dataset)
            history['val_loss'].append(val_loss)

            print(
                f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {epoch_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}'
            )

            if metrics:
                for metric in metrics:
                    metric_value = metric(val_outputs, val_labels)
                    if torch.is_tensor(metric_value):
                        metric_value = metric_value.detach().cpu().item()
                    history['metrics'][metric.__name__].append(metric_value)
                    print(f'{metric.__name__}: {metric_value:.4f}')

            model.train()


    return model, history