import os
import cv2
import torch
import timm
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


DATASET_PATH = 'C:/Users/Gabriel/TCC_GEMARS/dinov2/train/'
MODEL_OUTPUT = 'dinov2.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Usando dispositivo:")


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


print("Carregando DINOv2")
dinov2 = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0)
dinov2 = dinov2.to(DEVICE)
dinov2.eval()


classes_existentes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
print(f"🐟 Classes detectadas no diretório: {classes_existentes}")

X = []
y = []
classes_validas = []
novo_class_idx = 0


for class_name in classes_existentes:
    class_folder = os.path.join(DATASET_PATH, class_name)
    print(f"\n🔍 Processando classe [{class_name}]...")
    
    files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    classe_X = []
    classe_y = []
    
    for file in tqdm(files):
        img_path = os.path.join(class_folder, file)
        
       
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            img = None
            
        if img is None: 
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = dinov2(tensor_img).cpu().numpy().flatten()
            
        classe_X.append(embedding)
        classe_y.append(novo_class_idx)

   
    if len(classe_X) >= 2:
        X.extend(classe_X)
        y.extend(classe_y)
        classes_validas.append(class_name)
        novo_class_idx += 1
    else:
        print(f"⚠️ CLASSE IGNORADA: [{class_name}] não possui imagens válidas suficientes para validação (mínimo de 2).")

X = np.array(X)
y = np.array(y)


if len(X) == 0:
    print(" Erro fatal: Nenhuma imagem de nenhuma classe pôde ser carregada com sucesso.")
    exit()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTreinando SVM Multiclasse com {len(X_train)} frames...")
# probability=True permite extrair as porcentagens de certeza na inferência
clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
clf.fit(X_train, y_train)

# Avaliação
y_pred = clf.predict(X_test)
print("\n📊 --- MÉTRICAS DE DESEMPENHO MULTICLASSE ---")
print(f"Acurácia Geral: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=classes_validas))


dados_salvamento = {
    'classifier': clf,
    'classes': classes_validas
}
joblib.dump(dados_salvamento, MODEL_OUTPUT)
print(f"Salvo em: {MODEL_OUTPUT}")