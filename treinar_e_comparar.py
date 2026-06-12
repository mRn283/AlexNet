import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# FUNÇÃO PADRÃO DE TREINAMENTO
# ==========================================
def treinar_modelo(nome_modelo, modelo, criterion, optimizer, dataloader_treino, dataloader_val, dataset_treino, dataset_val, device, num_epochs=5):
    print(f"\nIniciando o treinamento da {nome_modelo} no {device.type.upper()}...")
    
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Total de parâmetros treináveis: {total_params:,}")
    
    tempo_inicial_total = time.time()
    
    for epoch in range(num_epochs):
        tempo_inicial_epoca = time.time()
        modelo.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloader_treino:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = modelo(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        tempo_final_epoca = time.time() - tempo_inicial_epoca
        epoch_loss = running_loss / len(dataset_treino)
        epoch_acc = running_corrects.double() / len(dataset_treino)
        
        print(f"Época {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f} | Acc Treino: {epoch_acc:.4f} | Tempo: {tempo_final_epoca:.2f}s")
    
    tempo_total = time.time() - tempo_inicial_total
    print(f"⏱️ Tempo total de treino da {nome_modelo}: {tempo_total:.2f} segundos")
    
    modelo.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modelo(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_corrects.double() / len(dataset_val)
    print(f"Acurácia Final de Validação ({nome_modelo}): {val_acc:.4f}")
    
    return total_params, tempo_total, val_acc.item()


if __name__ == '__main__':
    
    DATA_DIR = "./dados/cats_and_dogs_filtered"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VALID_DIR = os.path.join(DATA_DIR, "validation")

    transformacoes = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_treino_completo = datasets.ImageFolder(root=TRAIN_DIR, transform=transformacoes)
    dataset_validacao_completo = datasets.ImageFolder(root=VALID_DIR, transform=transformacoes)

    # 2. SELEÇÃO DO DATASET 
    QTD_TREINO = 2000       
    QTD_VALIDACAO = 1000    

    indices_treino = np.random.choice(len(dataset_treino_completo), QTD_TREINO, replace=False)
    indices_validacao = np.random.choice(len(dataset_validacao_completo), QTD_VALIDACAO, replace=False)

    dataset_treino = Subset(dataset_treino_completo, indices_treino)
    dataset_validacao = Subset(dataset_validacao_completo, indices_validacao)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ajustando dinamicamente os parâmetros de carregamento baseados no dispositivo encontrado
    if device.type == "cuda":
        BATCH_SIZE = 64
        num_workers_config = 2
        pin_memory_config = True
    else:
        BATCH_SIZE = 32
        num_workers_config = 0
        pin_memory_config = False

    dataloader_treino = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True, 
                                   num_workers=num_workers_config, pin_memory=pin_memory_config)
    dataloader_validacao = DataLoader(dataset_validacao, batch_size=BATCH_SIZE, shuffle=False, 
                                      num_workers=num_workers_config, pin_memory=pin_memory_config)

    print("📊 --- STATUS DO EXPERIMENTO ---")
    print(f"Imagens de treino utilizadas: {len(dataset_treino)}")
    print(f"Imagens de validação utilizadas: {len(dataset_validacao)}")
    print(f"🚀 Dispositivo de processamento ativo: {device.type.upper()}")
    if device.type == "cuda":
        print(f"Placa detectada: {torch.cuda.get_device_name(0)}")

    # CONFIGURAÇÃO E TREINO DA ALEXNET
    alexnet = models.alexnet(weights=None)
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 2)
    alexnet = alexnet.to(device)

    criterion_alexnet = nn.CrossEntropyLoss()
    optimizer_alexnet = optim.Adam(alexnet.parameters(), lr=0.001)

    params_alex, tempo_alex, acc_alex = treinar_modelo(
        "AlexNet", alexnet, criterion_alexnet, optimizer_alexnet, 
        dataloader_treino, dataloader_validacao, dataset_treino, dataset_validacao, device, num_epochs=5
    )

    # CONFIGURAÇÃO E TREINO DA RESNET-18
    resnet18 = models.resnet18(weights=None)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)
    resnet18 = resnet18.to(device)

    criterion_resnet = nn.CrossEntropyLoss()
    optimizer_resnet = optim.Adam(resnet18.parameters(), lr=0.001)

    params_res, tempo_res, acc_res = treinar_modelo(
        "ResNet-18", resnet18, criterion_resnet, optimizer_resnet, 
        dataloader_treino, dataloader_validacao, dataset_treino, dataset_validacao, device, num_epochs=5
    )

    # TABELA COMPARATIVA FINAL
    print("\n" + "="*60)
    print("📊 TABELA COMPARATIVA FINAL")
    print("="*60)
    print(f"AlexNet   -> Parâmetros: {params_alex:,} | Tempo Total: {tempo_alex:.2f}s | Acc Validação: {acc_alex:.4f}")
    print(f"ResNet-18 -> Parâmetros: {params_res:,} | Tempo Total: {tempo_res:.2f}s | Acc Validação: {acc_res:.4f}")
    print("="*60)