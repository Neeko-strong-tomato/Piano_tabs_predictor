import torch

def train(model, TrainDataLoader, ValDataLoader, criterion, optimizer, num_epochs, metrics=None):
    model.train()
    history = {'train_loss': [], 'val_loss': [], 'metrics': {metric.__name__: [] for metric in metrics} if metrics else {}}

    for epoch in range(num_epochs):

        epoch_loss = 0.0
        for inputs, labels in TrainDataLoader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
                    val_outputs = model(val_inputs)
                    v_loss = criterion(val_outputs, val_labels)
                    val_loss += v_loss.item() * val_inputs.size(0)
            val_loss /= len(ValDataLoader.dataset)
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
            if metrics:
                for metric in metrics:
                    metric_value = metric(val_outputs, val_labels)
                    print(f'{metric.__name__}: {metric_value:.4f}')
                    history['metrics'][metric.__name__].append(metric_value)
            model.train()

        else:
            # No validation data provided
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: N/A')
            if metrics:
                for metric in metrics:
                    metric_value = metric(outputs, labels)
                    print(f'{metric.__name__}: {metric_value:.4f}')
                    history['metrics'][metric.__name__].append(metric_value)

    return model, history