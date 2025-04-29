import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Triplet Loss for the 2D Exemplar Network
    
    As described in the paper, this uses a margin of 1.0
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

if __name__ == "__main__":
    # dummy embeddings
    batch_size = 8
    embedding_size = 1024
    
    anchor = torch.randn(batch_size, embedding_size)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_size)  # Similar to anchor
    negative = torch.randn(batch_size, embedding_size)  # Different from anchor
    
    # Initializing the loss function
    triplet_loss = TripletLoss(margin=1.0)
    
    # loss
    loss = triplet_loss(anchor, positive, negative)
    
    print("Triplet Loss Implementation:")
    print(f"Margin: {triplet_loss.margin}")
    print(f"Batch size: {batch_size}")
    print(f"Embedding size: {embedding_size}")
    print(f"Loss value: {loss.item()}")
    
    print("\nTriplet Loss successfully implemented!")
