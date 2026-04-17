# SB100 Agrônomo Virtual
## Pipeline de Extração e Classificação de Imagens

**Squad 02 Ingestão e Vetorização** | IC FAPESP | Versão 1.0 | Abril 2026

---

## 1. Visão Geral

Este script faz parte do pipeline de ingestão do projeto SB100 Agrônomo Virtual. Seu objetivo é extrair todas as figuras dos artigos científicos agronômicos já vetorizados, classificá-las automaticamente por tipo visual, e gerar um dataset estruturado para ser usado pela equipe de classificação.

---

## 2. Contexto no Pipeline SB100

| Squad | Responsabilidade | Relação com este script |
|---|---|---|
| Squad 01 | API de curadoria dos artigos | Fonte dos PDFs processados |
| Squad 02 (este) | Ingestão, vetorização e extração de imagens | Gera o dataset |
| Squad 03 | Banco de dados | Armazena e gerencia os dados gerados |
| Squad 04 | RAG / camada LLM | Consome vetores do Qdrant |
| Equipe de Classificação | Treinamento do classificador | Consome o dataset gerado aqui |

---

## 3. Estratégia Técnica

### 3.1 Por que renderizar páginas em vez de extrair por xref?

PDFs armazenam imagens como objetos binários (xref). Extrair por xref retorna fragmentos: logos de revista, watermarks e barras isoladas de gráfico armazenadas como objetos separados. O classificador recebia esses fragmentos sem contexto e classificava incorretamente.

A solução adotada foi **renderizar cada página completa via PyMuPDF** e recortar as regiões de figura via OpenCV. Isso garante que o classificador recebe a figura como o leitor humano a vê, inteira, com eixos, legendas visuais e contexto espacial.

### 3.2 Fluxo de processamento por página

| Etapa | Ferramenta | Descrição |
|---|---|---|
| 1. Renderização | PyMuPDF | Página renderizada em 150 DPI como imagem BGR |
| 2. Filtro de página | OpenCV (3 camadas) | Descarta páginas sem figuras reais |
| 3. Recorte de regiões | OpenCV contornos | Isola cada figura individualmente |
| 4. Filtro de crop | OpenCV (4 camadas) | Descarta recortes que são texto, não figura |
| 5. Busca de caption | PyMuPDF texto | Extrai legenda próxima à figura na página |
| 6. Classificação | CLIP zero-shot | Classifica o tipo visual da figura |
| 7. Persistência | Python / CSV | Salva imagem na pasta da categoria + metadata.csv |

### 3.3 Filtro de página com 3 camadas (`pagina_tem_figura`)

Idêntico ao filtro usado no Script 1 do pipeline principal. Descartou **552 páginas** de texto puro na última execução.

- **Camada 1, Cor HSV:** páginas com mais de 1.5% de pixels coloridos contêm figuras científicas
- **Camada 2, Fill density:** blobs P&B com densidade de preenchimento acima de 13% são candidatos
- **Camada 3, Direção de bordas Sobel:** ratio H/V acima de 2.0 indica texto; distribuído indica figura

### 3.4 Filtro de crop individual com 4 camadas (`crop_e_figura`)

Aplicado em cada recorte antes do CLIP. Descartou **1.277 recortes** de texto na última execução.

- **Camada 1, Cor:** mais de 1.2% de cor no crop indica figura e passa direto
- **Camada 2, Proporção:** ratio largura/altura acima de 5.0 indica parágrafo de texto e descarta
- **Camada 3, Linhas horizontais regulares (HoughLinesP):** mais de 10 linhas horizontais indica texto corrido e descarta
- **Camada 4, Bordas Sobel no crop:** ratio H/V acima de 2.5 indica texto e descarta

### 3.5 Extração de caption

Para cada figura, o script busca no texto da página um bloco que esteja espacialmente próximo (até 45pt, aproximadamente 1.5cm) à figura e que comece com palavra-chave de legenda em português, inglês ou espanhol (`Figura`, `Fig.`, `Gráfico`, `Figure`, `Chart`, `Tabla`, etc.).

A caption é salva na coluna `caption` do `metadata.csv`. Figuras sem legenda detectável ficam com caption vazio.

---

## 4. Categorias do Dataset

| Categoria | Pasta | Descrição |
|---|---|---|
| Gráfico de linhas | `grafico_linhas/` | Gráficos de tendência temporal com eixos X e Y |
| Gráfico de barras | `grafico_barras/` | Barras verticais ou horizontais comparando valores |
| Gráfico de pizza | `grafico_pizza/` | Gráficos de pizza ou donut mostrando proporções |
| Fotografia | `fotografia/` | Fotos de plantas, pragas, solo, campo agrícola |
| Diagrama | `diagrama/` | Diagramas científicos, ilustrações anatômicas, desenhos técnicos |
| Fluxograma | `fluxograma/` | Fluxogramas com setas conectando caixas ou etapas |
| Tabela como imagem | `tabela_imagem/` | Tabelas com linhas e colunas de dados |
| Mapa | `mapa/` | Mapas geográficos ou de distribuição espacial |
| Ícone | `icone/` | Ícones, logos, símbolos ou elementos decorativos |
| Outro | `outro/` | Imagens não classificadas ou ambíguas |

---

## 5. Estrutura de Pastas do Dataset

Localização no Google Drive:
```
MyDrive/PdfextractorHibrido/data/dataset_imagens/
```

Acesso direto: [dataset_imagens no Google Drive](https://drive.google.com/drive/folders/1Qq-Wjno3_V3_Gt4YjQB4ZqlDxG-CBJVE?usp=sharing)

Estrutura completa:
```
dataset_imagens/
├── grafico_linhas/
├── grafico_barras/
├── grafico_pizza/
├── fotografia/
├── diagrama/
├── fluxograma/
├── tabela_imagem/
├── mapa/
├── icone/
├── outro/
├── rejeitadas/
│   ├── muito_pequena/    ← recortes abaixo de 100x100px
│   ├── baixa_confianca/  ← CLIP com confiança abaixo de 22%
│   └── filtro_texto/     ← recortes identificados como texto pelo OpenCV
├── pdfs_base/            ← todos os PDFs usados para gerar este dataset
└── metadata.csv
```

A pasta `pdfs_base/` contém os 118 artigos científicos agronômicos processados. Eles estão disponíveis para que a equipe de classificação possa consultar o PDF original ao revisar uma imagem e confirmar se a classificação está correta.

---

## 6. Estrutura do metadata.csv

Cada linha corresponde a uma imagem, tanto as classificadas quanto as rejeitadas.

| Coluna | Tipo | Descrição |
|---|---|---|
| `pdf_origem` | string | Nome do arquivo PDF de origem (sem extensão) |
| `pagina` | int | Número da página no PDF onde a figura foi encontrada |
| `origem` | string | Sempre `renderizada` (estratégia de renderização de página) |
| `caption` | string | Texto da legenda extraído. Ex: `Figura 3. Produção de citros...` |
| `categoria` | string | Categoria classificada. Ex: `grafico_barras`, `filtro_texto` |
| `confianca` | float | Confiança do CLIP de 0.0 a 1.0. Vale 0.0 para rejeitadas por filtro |
| `largura` | int | Largura do recorte em pixels |
| `altura` | int | Altura do recorte em pixels |
| `path` | string | Caminho completo do arquivo de imagem salvo no Drive |

---

## 7. Resultados da Última Execução

| Métrica | Valor |
|---|---|
| PDFs processados | 118 |
| Tempo de execução | ~2min 35s (GPU CUDA) |
| Páginas descartadas por ser só texto | 552 |
| Crops descartados pelo filtro de texto | 1.277 |
| Imagens classificadas | 415 |
| Imagens com caption extraída | 23 de 1.693 (1.3%) |
| Rejeitadas por tamanho | 0 |
| Rejeitadas por baixa confiança | 1 |
| Total de registros no CSV | 1.693 |

### Distribuição por categoria

| Categoria | Quantidade | Observação |
|---|---|---|
| `grafico_barras` | 50 | Boa qualidade, categoria mais visualmente distinta |
| `tabela_imagem` | 58 | Pode conter ruído de tabelas de texto renderizadas |
| `icone` | 70 | Pode incluir logos coloridos de revista classificados incorretamente |
| `outro` | 135 | 32% do total, **principal alvo de revisão manual** |
| `grafico_linhas` | 34 | Boa qualidade |
| `fotografia` | 32 | Boa qualidade, texturas orgânicas bem detectadas |
| `mapa` | 15 | Boa qualidade |
| `diagrama` | 17 | Qualidade média, pode confundir com fluxograma |
| `fluxograma` | 4 | Baixo, o filtro de bordas descarta alguns casos |
| `grafico_pizza` | 0 | Não detectado, o CLIP tem dificuldade com este domínio |

---

## 8. Limitações Conhecidas

### 8.1 Classificação automática

O modelo CLIP foi treinado em imagens gerais da internet, não em figuras de artigos científicos agronômicos. As classificações são uma **aproximação inicial, não ground truth**. Categorias ambíguas como `diagrama` vs. `fluxograma` e `icone` vs. `outro` têm qualidade inferior.

### 8.2 Extração de caption

Apenas 1.3% das imagens têm caption extraída. A maioria das legendas nos PDFs do corpus não fica espacialmente adjacente à figura no arquivo digital porque o layout de duas colunas e o fluxo de texto dos artigos científicos frequentemente posiciona a legenda fora da margem de busca de 45pt.

### 8.3 Erros MuPDF

Mensagens `cmsOpenProfileFromMem failed` e `No common ancestor in structure tree` são erros de perfil de cor ICC em alguns PDFs. São **inofensivos**: o PyMuPDF ignora e continua o processamento normalmente.

---

## 9. Instruções para a Equipe de Classificação

### 9.1 Carregando o dataset

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Carrega só as categorias classificadas (ignora rejeitadas/)
dataset = ImageFolder(root="dataset_imagens/", transform=transform)
print(dataset.classes)  # lista as categorias
```

### 9.2 Usando o metadata.csv

```python
import pandas as pd

df = pd.read_csv("metadata.csv")

# Ver só as classificadas com alta confiança
df_boas = df[df["confianca"] > 0.50]

# Ver as rejeitadas pelo filtro de texto para revisão
df_texto = df[df["categoria"] == "filtro_texto"]

# Figuras com caption extraída
df_caption = df[df["caption"].str.len() > 0]
```

### 9.3 Revisão manual recomendada

Para obter um dataset de treino de qualidade, recomenda-se revisar manualmente:

- Toda a pasta `outro/` com 135 imagens, principal fonte de ruído
- Amostra de 20% da pasta `icone/`, que pode conter logos de revista classificados incorretamente
- Toda a pasta `rejeitadas/filtro_texto/`, que pode conter falsos negativos do filtro OpenCV
- Toda a pasta `rejeitadas/baixa_confianca/`, com imagens que o CLIP não soube classificar

### 9.4 Modelo recomendado para fine-tuning

| Modelo | Vantagem | Quando usar |
|---|---|---|
| EfficientNet-B0 | Leve, rápido, bom com dataset pequeno (~300 exemplos por classe) | Primeira iteração |
| ResNet-50 | Mais robusto, padrão da indústria | Dataset com mais de 500 exemplos por classe |
| ViT-small | Melhor para padrões globais de layout | Distinguir fluxograma de diagrama |

---

## 10. Dependências e Configuração

### 10.1 Instalação

```bash
pip install pymupdf transformers torch torchvision pillow tqdm opencv-python-headless
```

### 10.2 Parâmetros configuráveis

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `RENDER_DPI` | 150 | Resolução de renderização das páginas |
| `MIN_SIZE` | 100px | Tamanho mínimo do recorte em qualquer dimensão |
| `MIN_CONFIDENCE` | 0.22 | Confiança mínima do CLIP para aceitar classificação |
| `CAPTION_MARGIN` | 45pt (~1.5cm) | Distância máxima para buscar legenda na página |

### 10.3 Stack técnico

| Componente | Modelo / Biblioteca |
|---|---|
| Renderização PDF | PyMuPDF (fitz) |
| Filtro e recorte | OpenCV 4 |
| Busca de caption | PyMuPDF `get_text("dict")` |
| Classificação | CLIP `openai/clip-vit-base-patch32` via HuggingFace Transformers |
| Execução | Google Colab (GPU CUDA) |

---

## 11. Perguntas

### Sobre o dataset

**Por que só 415 imagens foram classificadas se o total é 1.693?**
Porque 1.277 recortes foram descartados pelo filtro de texto do OpenCV antes de chegar no CLIP, e o restante foi para as pastas de rejeitadas. As 415 são as que passaram por todos os filtros e receberam uma categoria.

**O que tem na pasta `outro/` com 135 imagens?**
Imagens que o CLIP não conseguiu encaixar com confiança em nenhuma categoria específica. Podem ser gráficos de dispersão, infográficos, equações renderizadas como imagem, ou figuras de domínio muito específico que o CLIP nunca viu durante o treinamento.

**Por que `grafico_pizza` zerou?**
O CLIP tem dificuldade em reconhecer gráficos de pizza em artigos científicos porque eles aparecem em escalas, cores e estilos muito variados. O modelo provavelmente classificou essas imagens como `outro` ou `diagrama`. A categoria existe no dataset para quando um modelo especializado for treinado.

**O metadata.csv é suficiente ou precisamos de banco de dados?**
Para o volume atual de 1.693 registros é suficiente. Qualquer ferramenta de ML lê CSV nativamente sem configuração extra. Se o dataset crescer para dezenas de milhares de imagens após novas rodadas, a migração para SQLite é simples porque as colunas já estão definidas.

**Por que só 1.3% das imagens têm caption?**
O layout de duas colunas dos artigos científicos frequentemente posiciona a legenda fora da margem de busca de 45pt que o script usa. A legenda pode estar na coluna oposta ou separada por outros elementos. É uma limitação estrutural dos PDFs do corpus, não um bug.

### Sobre o classificador

**Qual é a acurácia do classificador?**
Não temos esse número ainda porque não existe ground truth. As labels foram geradas automaticamente pelo CLIP e precisam de revisão manual para calcular acurácia real. O que sabemos é que `grafico_barras` e `grafico_linhas` têm qualidade boa visualmente, e `outro` e `icone` têm mais ruído.

**Por que usou CLIP e não um modelo específico para gráficos como ChartVLM?**
O CLIP foi escolhido como classificador inicial porque é leve, roda localmente sem custo e funciona zero-shot sem precisar de dados de treino. O ChartVLM seria mais preciso para gráficos mas não consegue classificar fotografias, diagramas e mapas. O objetivo aqui não era classificação perfeita mas gerar uma base inicial para a equipe de classificação refinar.

**Dá para treinar um modelo melhor com essas 1.693 imagens?**
Sim, mas só depois de revisar as labels manualmente. Se treinar com as labels do CLIP o modelo vai aprender os erros do CLIP. Com 400 a 500 imagens corretamente rotuladas já dá para fazer fine-tuning do EfficientNet-B0 com resultado bem melhor que o CLIP atual.

**Por que `diagrama` e `fluxograma` têm tão pouco?**
Duas razões. Primeiro, o corpus é predominantemente de artigos experimentais que têm mais gráficos de dados do que diagramas conceituais. Segundo, o filtro de bordas Sobel às vezes descarta fluxogramas porque eles têm muitos elementos com bordas horizontais parecidos com texto.

**O que significa confiança 0.22 do CLIP?**
É o threshold mínimo definido empiricamente. Abaixo disso o CLIP classificava com tanta incerteza que a categoria não tinha valor. 0.22 foi escolhido porque com valores menores o `outro` inflava muito. É um parâmetro ajustável no topo do script.

### Sobre a extração

**Por que renderizar a página inteira em vez de extrair as imagens diretamente do PDF?**
PDFs armazenam imagens como fragmentos separados. Uma barra de gráfico pode ser um objeto diferente da legenda do eixo. Extraindo por objeto o CLIP recebia fragmentos sem contexto. Renderizando a página e recortando via OpenCV o classificador vê a figura completa.

**Como o script sabe onde está a figura na página?**
OpenCV detecta contornos de regiões com contraste na página renderizada e calcula o bounding box de cada região. Regiões muito pequenas, muito finas ou que ocupam quase a página inteira são descartadas por filtros geométricos antes de chegar no CLIP.

**Os PDFs escaneados são tratados diferente?**
Não na versão atual. O pipeline renderiza todas as páginas independente de serem digitais ou escaneadas. A diferença é que páginas escaneadas não têm texto extraível, então a busca de caption retorna vazio para essas imagens.

**O que são os erros MuPDF que aparecem no log?**
São erros de perfil de cor ICC em alguns PDFs, mensagens `cmsOpenProfileFromMem failed` e `No common ancestor in structure tree`. São completamente inofensivos. O PyMuPDF ignora e continua processando normalmente. Aparecem em aproximadamente 18 dos 118 PDFs.

**Por que tem imagens na pasta `rejeitadas/filtro_texto/`?**
O filtro `crop_e_figura` descarta recortes com características de texto: muitas linhas horizontais paralelas, proporção muito larga, bordas predominantemente horizontais. Esses recortes são salvos lá em vez de descartados para que a equipe de classificação possa revisar e verificar se algum foi descartado incorretamente.

### Sobre o projeto

**Esse script faz parte do pipeline principal do SB100?**
É complementar, não parte do pipeline principal. O pipeline principal extrai texto e vetoriza no Qdrant. Este script roda separado sobre os mesmos PDFs já processados e gera o dataset de imagens para a equipe de classificação.

**Esse dataset vai ser atualizado quando chegarem novos PDFs?**
Sim. O script pode ser rodado novamente a qualquer momento sobre a pasta de PDFs concluídos. A célula 3.5 limpa o dataset anterior antes de rodar. Para manter versões diferentes basta renomear a pasta antes de rodar, por exemplo `dataset_imagens_v2`.

**Por que a pasta tem os PDFs originais junto com as imagens?**
Para facilitar a revisão manual. Quando a equipe de classificação encontrar uma imagem duvidosa, pode abrir o PDF original e ver a figura no contexto do artigo para decidir a label correta.

**O script roda em CPU também ou precisa de GPU?**
Roda nos dois. Com GPU CUDA levou 2min35s para 118 PDFs. Em CPU vai ser significativamente mais lento, provavelmente entre 20 e 40 minutos, porque o CLIP é mais pesado sem GPU. Os filtros OpenCV são leves e não dependem de GPU.

**Por que não usaram o Gemini para classificar as imagens como no Script 2 do pipeline?**
O Gemini tem limite de requisições e custo por chamada. Com 1.693 imagens isso geraria muitas chamadas de API e potencialmente custo. O CLIP roda localmente de graça e sem limite. Além disso um dos objetivos do projeto é explorar a substituição do Gemini por modelos locais, e este script faz parte dessa exploração.

---

*SB100 Agrônomo Virtual | Squad 02 Ingestão e Vetorização | IC FAPESP | 2026*
