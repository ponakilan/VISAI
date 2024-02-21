import torch
from sklearn.metrics import r2_score

def train_one_epoch(model, optimizer, criterion, dataloader, run, log_interval, device, save_path, epoch):
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

    torch.save(model, f'{save_path}/model_{epoch}.pt')
    return last_loss, model

def train(model, optimizer, criterion, train_dataloader, epochs, run, log_interval, device, save_path):
    for i in range(epochs):
        print(f"Epoch {i+1} started.")
        epoch_loss, model = train_one_epoch(model, optimizer, criterion, train_dataloader, run, log_interval, device, save_path, i)
        print(f"Epoch loss: {epoch_loss}")
    return model
