

import torch

from notebooks.train_mul_cnn import MultiChannelCNN
def load_trained_model(path_to_pth):
    checkpoint = torch.load(path_to_pth)
    
    # 1. Recover info
    class_names = checkpoint['class_names']
    input_channels = checkpoint['input_channels']
    
    # 2. Re-create the architecture
    loaded_model = MultiChannelCNN(num_classes=len(class_names), input_channels=input_channels)
    
    # 3. Load weights
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    return loaded_model, class_names

# usage: model, classes = load_trained_model("src/saved_models/mafulda_cnn_acc98.5_2023....pth")