from dataloader import load_2dimages
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn


#@def trains the model
#@param train_dataloader is the training dataset
#@param num_epochs is how many epochs we want to train for
#@param is th emodel we are training
#@param optimizer is the optimizer we are using
def train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)

            labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}')
        val_loss = validate(model, val_dataloader, criterion, epoch, num_epochs, device)
        if epoch % 5 == 0 and epoch != 0:
            save_checkpoint(model, running_loss, val_loss, epoch, optimizer, task)
            print(f"Checkpoint saved at epoch {epoch}")



def save_checkpoint(model, train_loss, val_loss, epoch, optimizer, task):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'train_loss': train_loss, 
        'val_loss': val_loss 
    }
    filepath = "/home/caleb/school/deep_learning/self-supervised-medical/src/2d/model_ckpt/" + task + "/" + "checkpoint" + str(epoch) + ".pth"
    torch.save(checkpoint, filepath)


def validate(model, val_dataloader, criterion, epoch, num_epochs, device):
    model.eval()
    running_loss = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        labels = torch.argmax(labels, dim=1)
        labels = labels.squeeze()
        loss = criterion(outputs, labels)

        running_loss += loss.item()
    
    running_loss = running_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {running_loss}')
    return running_loss






def main():
    
    model = models.densenet121(pretrained=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    task = input("What Task are we solving for (rotate): ")
    num_epochs = int(input("How many epochs we trying to do (paper recomends 20 for 2D tasks): "))

    print("Using device: " + str(device))

    if task == "rotate":
        #We are predicting four classes for the densenet rotation
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
    
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion = nn.CrossEntropyLoss()
    
    print("preprocessing images, this might take a moment ...")
    train_dataloader, val_dataloader = load_2dimages()
    print("preprocessing done, begining training ...")

    train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device)
    




if __name__ == "__main__":
    main()