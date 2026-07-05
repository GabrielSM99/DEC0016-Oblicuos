# Detecção e Contagem de Pesca Automatizado e Processamento em Lote 

## Introdução

Este projeto foi desenvolvido para dar suporte tecnológico às atividades de estudo ambiental e manejo sustentável da ONG GEMARS, situada em Torres - RS. O sistema consiste em uma solução de monitoramento automatizado e censo de peixes aplicada diretamente ao rolo mecânico de triagem e recolhimento de embarcações de pesca artesanal. 

A arquitetura emprega algoritmos avançados de visão computacional em tempo real baseados no framework **YOLOv26** integrado ao rastreador multialvo **ByteTrack**. O software gerencia de forma autônoma filas de processamento em lote, extrai evidências fotográficas panorâmicas em alta resolução através de um algoritmo de **Pico de Confiança (PCS)** e consolida inventários auditáveis. Visando a usabilidade e a estabilidade operacional para os biólogos da ONG, o sistema encapsula toda a complexidade matemática em uma interface gráfica web local responsiva desenvolvida em **Streamlit**.

> ⚠️ **Nota:** Na concepção original do projeto, previu-se a integração de um classificador híbrido composto pelo extrator de características latentes *Vision Transformer* **DINOv2** combinado a um classificador estatístico **SVM (Support Vector Machine)** para a identificação taxonômica fina das espécies. Entretanto, devido ao fato de o dataset de imagens taxonômicas ainda não estar completo, o módulo do DINOv2 foi temporariamente omitido desta versão piloto. O sistema atual foca puramente no censo volumétrico, na contagem absoluta de indivíduos e no rastreamento geométrico estável.

---

## Descrição do Projeto

O software foi projetado para operar no cenário real das atividades de campo da ONG GEMARS, onde os analistas frequentemente precisam processar múltiplos arquivos de vídeo pesados (frequentemente superiores a 1 GB cada) gravados pelas câmeras instaladas nos conveses dos barcos. 

A abordagem de design substitui os tradicionais scripts de console ou binários compactados instáveis por uma aplicação distribuída localmente que roda de forma nativa no navegador do usuário. O sistema realiza a leitura dos arquivos de mídia, decodifica as matrizes de imagem quadro a quadro e gera uma estrutura de diretórios auto-organizada para mitigar falhas humanas na catalogação dos dados.

### Objetivos do Sistema

- **Censo Volumétrico Absoluto:** Realizar a contagem automatizada e precisa de indivíduos que passam pelo rolo de tração mecânica.
- **Processamento Assíncrono em Lote:** Permitir o upload simultâneo de múltiplos vídeos, gerenciando uma fila de execução sequencial de forma estável.
- **Evidenciamento Contextual Autônomo:** Capturar frames panorâmicos (tela cheia) contendo as marcações visuais das caixas delimitadoras nos instantes exatos de maior nitidez física de cada animal.
- **Mapeamento Temporal Científico:** Registrar o frame e o horário exato (Minutos:Segundos) de aparição de cada espécime para fins de analise biológica e estatística.

### Visão de Alto Nível da Pipeline

O pipeline de processamento do sistema segue o fluxo sequencial abaixo:

```text
[Vídeos de Entrada (.mp4)] ──> [Streamlit Web UI] ──> [Fila de Processamento em Lote]
                                                                │
  ┌─────────────────────────────────────────────────────────────┘
  ▼
[Leitura de Frame (OpenCV)] ──> [Detecção YOLOv26] ──> [Rastreamento de ID (ByteTrack)]
                                                                │
  ┌─────────────────────────────────────────────────────────────┘
  ▼
[Filtro PCS (Peak Confidence)] ──> Peixe saiu do frame?
                                          │
        ┌─────────────────────────────────┴────────────────────────┐
        ▼ Sim                                                      ▼ Não
[Desenhar Bounding Box + ID]                               [Atualizar Centroide]
        │                                                          │
        ├──> Salvar Frame Panorâmico em /capturas                  └──> Próximo Frame
        ├──> Incrementar Contador de Peixes
        └──> Gravar Registro com Horário (MM:SS) no DataFrame
                                │
  ┌─────────────────────────────┘
  ▼
[Geração da Pasta de Saída com o Nome do Vídeo]
├── inventario_[nome].xlsx
├── output_[nome].mp4
└── /capturas/peixe_nXXX_idX_completo.jpg

```

### Estrutura de Saída de Dados (Outputs)

Ao finalizar o processamento de um lote de vídeos, o sistema cria pastas individuais no diretório raiz do programa. Se o usuário submeter um vídeo chamado video_ancoradouro01.mp4, o sistema gerará a seguinte árvore estrutural:

```text

video_hora01/
├── inventario_video_hora01.xlsx  # Planilha Excel contendo os registros consolidados
├── output_video_hora01.mp4       # Cópia técnica do vídeo com as marcações visuais gravadas
└── capturas/                            # Imagens panorâmicas com as bounding boxes desenhadas
    ├── peixe_n001_id4_completo.jpg      # Evidência fotográfica do peixe #1 no pico de certeza
    ├── peixe_n002_id7_completo.jpg      # Evidência fotográfica do peixe #2 no pico de certeza
    └── ...

```

## Componentes da Planilha de Inventário (.xlsx)

Cada linha do arquivo Excel exportado corresponde a uma evidência irrefutável e possui as seguintes colunas:

- Nº Inventário: Índice incremental do espécime na amostragem (ex: #001).

- ID Rastreio (YOLO): Identificador persistente atribuído pelo ByteTrack.

- Horário de Passagem (MM:SS): Momento exato em que o indivíduo cruzou o rolo mecânico, calculado matematicamente com base no FPS do vídeo.

- Frame do Pico: Número do quadro exato onde ocorreu o ápice de confiança morfológica.

- Confiança Máxima da IA: O maior score percentual de certeza obtido pela YOLOv26 no ciclo de vida do objeto.

- Nome do Arquivo de Imagem: Ponteiro de texto para correlacionar a linha à foto presente na pasta de capturas.

## Configuração do Ambiente e Software:

O sistema foi inteiramente projetado em ambiente Linux (Pop!_OS) e portado para operação nativa no Microsoft Windows 10/11 através de interpretadores Python 3.11, garantindo cross-plataforma e isolamento de dependências via ambientes virtuais (venv).

### Instalação e Preparação 

- Clonar/Baixar o Repositório: Extraia o código-fonte em uma pasta local no computador (ex: C:\Sistema_DEC0021).

- Download dos Arquivos Gigantes (Nuvem Externa): Devido às diretrizes de armazenamento do GitHub (limite de 100 MB por push), os arquivos binários pesados de dados não são versionados diretamente no Git. Baixe-os através dos links homologados e insira-os na raiz do projeto:

* Banco de Vídeos de Amostragem (1 GB+ cada): [https://drive.google.com/drive/folders/1SF3d83O2An9lseeRjOQ2-PUvBmpsO2UJ?usp=sharing] 

* Instalação dos Módulos via Terminal: Abra o terminal (PowerShell ou Prompt de Comando) na raiz do projeto e execute os comandos abaixo para inicializar o ambiente virtual e instalar as bibliotecas necessarias:

### Criar o diretório isolado do ambiente virtual
python -m venv venv

### Liberar permissão de execução de scripts locais no Windows PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

### Ativar o ambiente virtual
.\venv\Scripts\Activate.ps1

### Atualizar o gerenciador de pacotes e instalar as bibliotecas do ecossistema
pip install streamlit ultralytics opencv-python pandas openpyxl torch torchvision

## Configuração de Upload do Servidor (config.toml)

Como a aplicação processa arquivos de vídeo de alta definição que ultrapassam as limitações padrão de requisição de navegadores, é obrigatório manter o arquivo de configuração estática na raiz do projeto em .streamlit/config.toml preenchido com os seguintes parâmetros de alocação de memória:

[server]
maxUploadSize = 5000

* Este parâmetro expande a capacidade de upload da interface web local para receber arquivos de até 5 Gigabytes de forma estável.

## Operação do Sistema:

Para facilitar o uso por profissionais de outras áreas, como biólogos, a rotina de inicialização foi totalmente automatizada.

Inicialização Rápida (Interface Gráfica):


- 1. Vá até a pasta do projeto através do Explorador de Arquivos do Windows.

- 2. Dê dois cliques no arquivo Iniciar_Sistema.bat.

- 3. O script em lote executará os bastidores do prompt de comando, ativará a venv, lerá os arquivos de configuração e abrirá de forma automática uma aba no seu navegador web padrão (Chrome, Edge ou Firefox) apontando para o endereço local http://localhost:8501.

## Utilização do Sistema Web:

- 1. Fila de Upload: Clique no campo central da página web ou arraste múltiplos arquivos de vídeo simultaneamente de dentro das suas pastas para a zona de seleção do Streamlit.

- 2. Processamento: Clique no botão destacado "Iniciar Processamento de Inventário em Lote".

- 3. Acompanhamento: O sistema exibirá uma barra de progresso percentual dinâmica e caixas de texto atualizadas em tempo real informando qual quadro está sendo processado para cada vídeo da fila.

- 4. Fechamento: Ao concluir todos os arquivos, uma animação de sucesso será exibida na tela e as métricas totais de contagem por vídeo serão plotadas no painel. Os relatórios físicos estarão salvos e organizados nas respectivas pastas locais.

## Estrutura de Arquivos do Repositório

A organização dos diretórios foi estruturada seguindo as boas práticas de Engenharia de Software, mantendo o versionamento leve e isolando os artefatos gerados:

```text
├── .streamlit/
│   └── config.toml             # Arquivo de configuração de limites do Streamlit
├── venv/                       # Diretório do ambiente virtual (Ignorado no Git via .gitignore)
├── app_gemars.py               # Script principal contendo a interface Web e a pipeline de visão
├── bestyolo26.pt               # Pesos da rede YOLOv26 treinada
├── Iniciar_Sistema.bat         # Arquivo de lote para execução automatizada em um clique
├── .gitignore                  # Filtro do Git para impedir upload de vídeos e arquivos pesados
└── README.md                   # Esta documentação completa do sistema
```

## Tecnologias Utilizadas:

- Python 3.11: Linguagem base de alto nível para desenvolvimento ágil.

- Ultralytics YOLOv26 & ByteTrack: Algoritmo de detecção de objetos e rastreamento robusto por centroides bidimensionais.

- OpenCV (Open Source Computer Vision Library): Manipulação de imagens, processamento de streams e gravação de fluxos de vídeo compactados.

- Streamlit: Framework de UI moderno e reativo para desenvolvimento de interfaces de video.

- Pandas & OpenPyXL: Manipulação de estruturas de dados e exportação limpa de matrizes para planilhas eletrônicas padronizadas do Microsoft Excel.

- PyTorch: Framework de aprendizado profundo atuando como backend para processamento matemático acelerado por hardware (CUDA/CPU).

Este software foi desenvolvido como uma solução de Engenharia de Computação com foco em impacto ecológico, transparência em auditorias ambientais e automação científica para a preservação marinha.

 * GABRIEL SODRE DE MOURA - gabrielsodredev@gmail.com
