from torch.optim import Adam
import torch.nn as nn
import copy
import torch

from models import create_model
from data import get_dataloaders, main_transformations

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.to(device)
    min_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # deep copy the model if better accuracy found in validation
            if phase == 'val':
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc = epoch_acc
                    counter = 0
                else:
                    counter += 1
                    if counter == patience:
                        print(f"Early stopping triggered after epoch {epoch + 1} due to no improvement in validation loss.")
                        model.load_state_dict(best_model_wts)
                        return model, best_acc

    model.load_state_dict(best_model_wts)
    return model, best_acc

def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()  

    correct = 0
    total = 0

    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy

def evaluate_model(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    # No gradient is needed for evaluation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_preds, all_labels


def main():
    transformations = main_transformations()
    train_loader, val_loader, test_loader = get_dataloaders(transformations)
    models_dict = {
    'resnet101': create_model('resnet101', 4),
    'vgg16': create_model('vgg16', 4),
    'swin_transformer': create_model('swin_transformer', 4)
}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    for model_name, model in models_dict.items():
        print(f"Training {model_name}...")
        optimizer = Adam(model.parameters(), lr=1e-4)
        trained_model, best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=device,patience=5)
        torch.save(trained_model.state_dict(), f'{model_name}_best_weights.pth')
        print(f"Best validation accuracy for {model_name}: {best_acc:.4f}")

    for model_name, _ in models_dict.items():
        model = create_model(model_name, num_classes=4)  
        model.load_state_dict(torch.load(f'{model_name}_best_weights.pth'))
        model.to(device)
        print(f"Testing {model_name}...")
        test_acc = test_model(model, test_loader, device=device)

        predictions, true_labels = evaluate_model(model, test_loader, device)
        conf_matrix = confusion_matrix(true_labels, predictions)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'{model_name}_confusion_matrix.png')

        print(test_acc)
    

if __name__ == "__main__":
    main()