import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Caminho das pastas filtradas
DATA_DIR = "./dados/cats_and_dogs_filtered"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "validation")

# 2. Transformações (Pré-processamento obrigatório para a AlexNet)
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Padrão do ImageNet
                         std=[0.229, 0.224, 0.225])
])

# 3. Carregar os datasets (o ImageFolder identifica as subpastas como as classes)
dataset_treino = datasets.ImageFolder(root=TRAIN_DIR, transform=transformacoes)
dataset_validacao = datasets.ImageFolder(root=VALID_DIR, transform=transformacoes)

# 4. Criar os DataLoaders (Lotes de 32 imagens para não sobrecarregar seu CPU)
BATCH_SIZE = 32
dataloader_treino = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
dataloader_validacao = DataLoader(dataset_validacao, batch_size=BATCH_SIZE, shuffle=False)

# --- TESTE DE VERIFICAÇÃO ---
print("\n --- STATUS DO DATASET ---")
print(f"Imagens de treino encontradas: {len(dataset_treino)}")
print(f"Imagens de validação encontradas: {len(dataset_validacao)}")
print(f"Classes mapeadas: {dataset_treino.classes} (onde {dataset_treino.class_to_idx})")

# Testando se o PyTorch consegue ler um lote com sucesso
imagens, labels = next(iter(dataloader_treino))
print(f"Formato do lote de imagens: {imagens.shape} -> [Lote, Canais RGB, Altura, Largura]")
print(" Tudo funcionando perfeitamente!")