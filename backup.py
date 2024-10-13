def train_and_validate_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, device, train_df, validation_df, patience, min_delta, scheduler):
    best_validation_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()  # Clear the GPU cache at the start of each epoch

        # Training phase
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc='Training', leave=False)

        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_progress_bar.set_postfix({'Loss': f'{loss.item():.10f}'})  # Update tqdm with current loss

        epoch_loss = running_loss / len(train_df)
        print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.10f}')

        # Validation phase
        model.eval()
        validation_loss = 0.0
        correct = 0
        validation_progress_bar = tqdm(validation_loader, desc='Validation', leave=False)

        with torch.no_grad():
            for inputs, labels in validation_progress_bar:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct += (predicted == labels).sum().item()

                validation_progress_bar.set_postfix({'Loss': f'{loss.item():.10f}'})  # Update tqdm with current loss

        validation_loss = validation_loss / len(validation_df)
        validation_accuracy = correct / len(validation_df)
        print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {validation_loss:.10f}, Validation Accuracy: {validation_accuracy:.10f}')

        scheduler.step(validation_loss)
        
        # Early stopping check
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            print(f"Validation loss improved. New best validation loss: {best_validation_loss:.10f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

num_epochs = 50
patience = 3  # Early stopping patience
min_delta = 0.001
best_validation_loss = float('inf')
epochs_no_improve = 0
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
train_and_validate_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, device, train_df, validation_df, patience, min_delta, scheduler)