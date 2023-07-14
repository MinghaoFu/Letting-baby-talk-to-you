import torch.nn as nn

class cross_entropy_loss(nn.Module): 
    def __init__(self, args):
        super(cross_entropy_loss, self).__init__()
        if args.reweight:
            self.criterion = nn.CrossEntropyLoss(weight=args.cls_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.args = args
        
    def forward(self, projections, targets):
        loss = self.criterion(projections, targets)
        return loss