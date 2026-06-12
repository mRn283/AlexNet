import os
import zipfile
import urllib.request

# URL do dataset reduzido (68 MB)
url = "https://dl.google.com/mlcc/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = "cats_and_dogs_filtered.zip"
extract_path = "./dados"

print("-> Baixando o dataset reduzido de Gatos e Cachorros (68MB)...")

# Criamos uma requisição fingindo ser um navegador comum para evitar o Erro 403
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
)

# Baixa o arquivo em blocos de 1MB para não sobrecarregar a memória
with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
    while True:
        chunk = response.read(1024 * 1024) # 1 Megabyte
        if not chunk:
            break
        out_file.write(chunk)

print("-> Extraindo os arquivos...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Remove o arquivo .zip para limpar espaço
os.remove(zip_path)

print("-> Tudo pronto! Seus dados estão salvos na pasta './dados/cats_and_dogs_filtered'")