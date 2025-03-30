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
def train(train_dataloader, num_epochs, model, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for inputs, labels in train_dataloader:
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



def main():
    
    model = models.densenet121(pretrained=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = input("What Task are we solving for: ")
    num_epochs = int(input("How many epochs we trying to do (paper recomends 20 for 2D tasks): "))

    print("Using device: " + str(device))

    if task == "rotate":
        #We are predicting four classes for the densenet rotation
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion = nn.CrossEntropyLoss()
    
    print("preprocessing images, this might take a moment ...")
    train_dataloader = load_2dimages()
    print("preprocessing done, begining training ...")

    train(train_dataloader, num_epochs, model, optimizer, criterion)
    




if __name__ == "__main__":
    main()