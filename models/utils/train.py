import torch

def train_one_epoch(model, optimizer, criterion, dataloader, run, log_interval, epoch, device):
    model.train()
    running_loss = 0.0
    last_loss = 0.0
    
    for i, data in enumerate(dataloader):
        sequences, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets.reshape(outputs.shape))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_interval == log_interval - 1:
            last_loss = running_loss/log_interval
            run.log({"loss": last_loss})
            running_loss = 0.0
            torch.save(model, f'model_{epoch}_{i}.pt')

    return last_loss

def train(model, optimizer, criterion, train_dataloader, epochs, run, log_interval, device):
    for i in range(epochs):
        print(f"Epoch {i+1} started.")
        epoch_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, run, log_interval,i,  device)
        print(f"Epoch loss: {epoch_loss}")
