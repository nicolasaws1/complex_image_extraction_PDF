# ============================================================
# SB100 — Extração, Caption e Classificação de Imagens
# Squad 02 — Ingestão e Vetorização
# Gera dataset classificado para treinamento de modelo ML
#
# Estratégia:
#   Renderiza cada página completa via PyMuPDF → filtra com
#   pagina_tem_figura() (OpenCV 3 camadas, idêntico ao Script 1)
#   → recorta regiões de figura via contornos OpenCV → busca
#   caption no texto da página → classifica com CLIP.
#
#   Por que renderizar em vez de extrair por xref?
#   Extração por xref retorna fragmentos (logos, watermarks,
#   barras isoladas). Renderizar garante que o CLIP recebe a
#   figura completa, exatamente como o leitor humano vê.
# ============================================================

# ==============================================================
# CÉLULA 1 — Instalação de dependências
# ==============================================================
# !pip install pymupdf transformers torch torchvision pillow tqdm opencv-python-headless -q

# ==============================================================
# CÉLULA 2 — Imports e mount do Drive
# ==============================================================
import fitz  # PyMuPDF
import io
import os
import csv
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm.notebook import tqdm

from google.colab import drive
drive.mount('/content/drive')

# ==============================================================
# CÉLULA 3 — Configurações globais
# ==============================================================
PDF_DIR     = "/content/drive/MyDrive/PdfextractorHibrido/data/PDFs concluídos"
DATASET_DIR = "/content/drive/MyDrive/PdfextractorHibrido/data/dataset_imagens"

RENDER_DPI      = 150    # resolução de renderização (mesma do Script 1)
MIN_SIZE        = 100    # px — descarta recortes menores que isso em qualquer dimensão
MIN_CONFIDENCE  = 0.22   # confiança mínima CLIP — abaixo vai pra rejeitadas/baixa_confianca
CAPTION_MARGIN  = 45.0   # pontos PDF de distância máxima para buscar caption (~1.5 cm)

# Categorias: chave = nome da pasta, valor = prompt CLIP em inglês
CATEGORIES = {
    "grafico_linhas":  "a line chart or line graph showing data trends over time with X and Y axes",
    "grafico_barras":  "a bar chart or bar graph with vertical or horizontal bars comparing values",
    "grafico_pizza":   "a pie chart or donut chart showing percentages or proportions",
    "fotografia":      "a photograph of plants, crops, insects, soil, field or agricultural scenery",
    "diagrama":        "a scientific diagram, anatomical illustration or labeled technical drawing",
    "fluxograma":      "a flowchart or process flow diagram with arrows connecting boxes or steps",
    "tabela_imagem":   "a table or data grid with rows and columns of numbers or text",
    "mapa":            "a geographical map, spatial distribution map or choropleth map",
    "icone":           "a small icon, logo, symbol or decorative graphic element",
    "outro":           "unclassified or unclear image content",
}

CATEGORY_LABELS = list(CATEGORIES.keys())
CATEGORY_TEXTS  = list(CATEGORIES.values())

# Palavras-chave que iniciam uma legenda/caption (PT + EN + ES)
CAPTION_KEYWORDS = {
    "figura", "fig.", "fig ",  "gráfico", "grafico",  "tabela",
    "imagem", "foto", "mapa",  "diagrama", "quadro",   "esquema",
    "ilustração", "ilustracao",
    "figure", "chart", "graph", "table",   "image",    "photo",
    "map",    "diagram","scheme","panel",
    "tabla",  "imagen",
}

# ==============================================================
# CÉLULA 3.5 — Limpar dataset anterior (rodar uma vez)
# ==============================================================
import shutil

if Path(DATASET_DIR).exists():
    shutil.rmtree(DATASET_DIR)
    print(f"🗑️  Dataset anterior removido: {DATASET_DIR}")
else:
    print("ℹ️  Nenhum dataset anterior encontrado.")
    
# ==============================================================
# CÉLULA 4 — Criar estrutura de pastas do dataset
# ==============================================================
def create_dataset_structure(base_dir: str):
    for cat in CATEGORY_LABELS:
        Path(base_dir, cat).mkdir(parents=True, exist_ok=True)
    Path(base_dir, "rejeitadas", "muito_pequena").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "rejeitadas", "baixa_confianca").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "rejeitadas", "filtro_texto").mkdir(parents=True, exist_ok=True)
    print(f"✅ Estrutura de pastas criada em: {base_dir}")

create_dataset_structure(DATASET_DIR)

# ==============================================================
# CÉLULA 5 — Carregar modelo CLIP
# ==============================================================
print("⏳ Carregando CLIP (primeira vez pode demorar ~1 min)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print(f"✅ CLIP carregado em: {device}")

# ==============================================================
# CÉLULA 6 — Filtro de página (idêntico ao Script 1)
# ==============================================================

def pagina_tem_figura(img_bgr: np.ndarray) -> bool:
    """
    Decide se uma página renderizada contém figuras reais
    (gráficos, fotos, diagramas) ou é só texto/logos simples.

    3 camadas (mesma lógica do Script 1 do pipeline):
      1) COR     — figuras científicas quase sempre têm cor
      2) DENSITY — texto é esparso; figuras são densas
      3) BORDAS  — texto tem bordas horizontais; figuras, multidirecionais
    """
    h, w       = img_bgr.shape[:2]
    area_total = h * w

    # ── CAMADA 1: COR
    hsv          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mascara_cor  = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 240)
    pct_colorido = np.sum(mascara_cor) / area_total
    if pct_colorido > 0.015:
        return True

    # ── CAMADA 2 + 3: BLOCOS P&B
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dil    = cv2.dilate(thr, kernel, iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (bw * bh) / area_total < 0.03:
            continue

        roi    = thr[y:y+bh, x:x+bw]
        filled = np.sum(roi > 0) / (bw * bh)
        if filled < 0.13:
            continue

        sob_h  = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        sob_v  = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        e_h    = np.sum(np.abs(sob_h))
        e_v    = np.sum(np.abs(sob_v))
        ratio  = (e_h / e_v) if e_v > 0 else 999
        if ratio > 2.0:
            continue
        return True

    return False


# ==============================================================
# CÉLULA 6.5 — Filtro de crop individual
# ==============================================================

def crop_e_figura(crop_bgr: np.ndarray) -> bool:
    """
    Decide se um recorte individual é uma figura real ou texto.

    O pagina_tem_figura() garante que A PÁGINA tem figura, mas o
    recortar_regioes() corta tudo com contraste — incluindo blocos
    de texto, equações e caixas de abstract na mesma página.

    Este filtro roda em cada crop antes do CLIP com 4 camadas:

    1) COR: qualquer cor real → figura (mesma lógica da página,
       threshold mais baixo porque o crop é menor)
    2) PROPORÇÃO: crops muito largos e baixos são linhas de texto
    3) LINHAS HORIZONTAIS REGULARES: padrão característico de
       parágrafos — detect via HoughLinesP
    4) BORDAS: ratio H/V alto → texto; distribuído → figura
       (aplicado diretamente no crop, sem precisar de blob)
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return False

    # ── CAMADA 1: COR
    hsv         = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mascara_cor = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 240)
    pct_cor     = np.sum(mascara_cor) / (h * w)
    if pct_cor > 0.012:          # >1.2% de cor → figura
        return True

    gray   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # ── CAMADA 2: PROPORÇÃO — parágrafos são muito mais largos que altos
    aspect = w / h
    if aspect > 5.0:             # muito horizontal → provavelmente texto
        return False

    # ── CAMADA 3: LINHAS HORIZONTAIS REGULARES (padrão de parágrafo)
    edges  = cv2.Canny(gray, 50, 150)
    linhas = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=w * 0.35, maxLineGap=8
    )
    if linhas is not None:
        # Conta linhas quase perfeitamente horizontais (ângulo < 5°)
        horizontais = sum(
            1 for l in linhas
            if abs(l[0][3] - l[0][1]) < 5   # delta_y pequeno
        )
        # Muitas linhas horizontais regularmente espaçadas → texto
        if horizontais > 10:
            return False

    # ── CAMADA 4: DIREÇÃO DE BORDAS no crop inteiro
    sob_h = cv2.Sobel(thr, cv2.CV_64F, 0, 1, ksize=3)
    sob_v = cv2.Sobel(thr, cv2.CV_64F, 1, 0, ksize=3)
    e_h   = np.sum(np.abs(sob_h))
    e_v   = np.sum(np.abs(sob_v))
    ratio = (e_h / e_v) if e_v > 0 else 999

    if ratio > 2.5:              # dominado por bordas horizontais → texto
        return False

    return True


# ==============================================================
# CÉLULA 7 — Recorte de regiões de figura (OpenCV)
# ==============================================================

def recortar_regioes(img_bgr: np.ndarray):
    """
    Recebe a imagem BGR da página já renderizada.
    Retorna lista de (crop_bgr, x, y, w, h) em pixels —
    cada item é uma região candidata a figura.

    A lógica de dilatação e filtros é mais agressiva que o
    pagina_tem_figura(): aqui queremos isolar cada figura
    individualmente, não apenas detectar a presença de uma.
    """
    page_h, page_w = img_bgr.shape[:2]
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thr  = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Dilatação mais suave que o filtro de página para não fundir figuras adjacentes
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(thr, kernel, iterations=3)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regioes = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        # Descarta regiões que cobrem quase a página inteira (fundo ou bloco de texto)
        if w > page_w * 0.90 or h > page_h * 0.90:
            continue
        # Descarta por tamanho mínimo em pixels
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        # Descarta proporções absurdas (linhas de cabeçalho/rodapé)
        aspect = w / h
        if aspect > 8 or aspect < 0.12:
            continue

        crop = img_bgr[y:y+h, x:x+w].copy()
        regioes.append((crop, x, y, w, h))

    return regioes


# ==============================================================
# CÉLULA 8 — Busca de caption na página
# ==============================================================

def find_caption(page, img_rect_pdf: fitz.Rect, margin: float = CAPTION_MARGIN) -> str:
    """
    Busca a legenda (caption) da figura na página PDF.

    Recebe img_rect_pdf em coordenadas de pontos PDF
    (convertido de pixels → pontos antes de chamar esta função).

    Estratégia:
      1. Extrai blocos de texto com bounding boxes da página.
      2. Filtra blocos que começam com palavra-chave de legenda (PT/EN/ES).
      3. Verifica proximidade vertical (acima ou abaixo) dentro de `margin`.
      4. Verifica sobreposição horizontal (tolerância de 20pt).
      5. Retorna o bloco mais próximo, ou "" se não encontrado.
    """
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except Exception:
        return ""

    best_caption = ""
    best_dist    = float("inf")

    for block in blocks:
        if block.get("type") != 0:
            continue

        block_text = " ".join(
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()

        if len(block_text) < 5:
            continue

        lower      = block_text.lower()
        is_caption = any(lower.startswith(kw) for kw in CAPTION_KEYWORDS)
        if not is_caption:
            continue

        b_rect     = fitz.Rect(block["bbox"])
        dist_below = b_rect.y0 - img_rect_pdf.y1
        dist_above = img_rect_pdf.y0 - b_rect.y1
        h_overlap  = min(img_rect_pdf.x1, b_rect.x1) - max(img_rect_pdf.x0, b_rect.x0)

        if h_overlap >= -20:
            if 0 <= dist_below <= margin and dist_below < best_dist:
                best_dist    = dist_below
                best_caption = block_text
            elif 0 <= dist_above <= margin and dist_above < best_dist:
                best_dist    = dist_above
                best_caption = block_text

    return " ".join(best_caption.split())[:300]


# ==============================================================
# CÉLULA 9 — Funções auxiliares
# ==============================================================

def classify_image(pil_image: Image.Image):
    """Classifica imagem PIL com CLIP zero-shot. Retorna (categoria, confiança)."""
    inputs = clip_processor(
        text=CATEGORY_TEXTS,
        images=pil_image,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1).squeeze()
    best_idx = probs.argmax().item()
    return CATEGORY_LABELS[best_idx], probs[best_idx].item()


def make_filename(pdf_stem: str, page_num: int, idx: int, pil_image: Image.Image) -> str:
    """Nome único baseado em hash MD5 do conteúdo da imagem."""
    buf      = io.BytesIO()
    pil_image.save(buf, format="PNG")
    h        = hashlib.md5(buf.getvalue()).hexdigest()[:8]
    safe     = pdf_stem[:35].replace(" ", "_").replace("/", "_")
    return f"{safe}_p{page_num:04d}_i{idx:03d}_{h}.png"


def pixels_to_pdf_rect(x, y, w, h, scale: float) -> fitz.Rect:
    """
    Converte coordenadas de pixels (OpenCV) para pontos PDF (fitz.Rect).
    scale = RENDER_DPI / 72  →  ponto_pdf = pixel / scale
    """
    return fitz.Rect(x / scale, y / scale,
                     (x + w) / scale, (y + h) / scale)


# ==============================================================
# CÉLULA 10 — Pipeline principal
# ==============================================================

pdf_files = sorted(Path(PDF_DIR).rglob("*.pdf"))
print(f"📂 PDFs encontrados: {len(pdf_files)}\n")

metadata_rows = []
stats = {cat: 0 for cat in CATEGORY_LABELS}
stats.update({"rejeitada_tamanho": 0, "rejeitada_confianca": 0,
              "paginas_sem_figura": 0, "crop_texto": 0, "erros": 0})

SCALE = RENDER_DPI / 72.0   # fator de escala pixels ↔ pontos PDF

for pdf_path in tqdm(pdf_files, desc="PDFs"):
    pdf_stem = pdf_path.stem
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"❌ Erro ao abrir {pdf_path.name}: {e}")
        stats["erros"] += 1
        continue

    for page_num, page in enumerate(doc):
        # ── 1. Renderiza a página completa ──────────────────────────────
        try:
            mat       = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
            pix       = page.get_pixmap(matrix=mat, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8
                        ).reshape(pix.height, pix.width, 3)
            img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  ⚠️  Renderização falhou p.{page_num+1}: {e}")
            stats["erros"] += 1
            continue

        # ── 2. Filtro de página (3 camadas — idêntico ao Script 1) ──────
        if not pagina_tem_figura(img_bgr):
            stats["paginas_sem_figura"] += 1
            continue

        # ── 3. Recorta regiões de figura via OpenCV ──────────────────────
        regioes = recortar_regioes(img_bgr)
        if not regioes:
            stats["paginas_sem_figura"] += 1
            continue

        for idx, (crop_bgr, cx, cy, cw, ch) in enumerate(regioes):
            # ── 4. Filtro individual: crop é figura ou texto? ────────────
            if not crop_e_figura(crop_bgr):
                stats["crop_texto"] += 1
                # Salva em rejeitadas/filtro_texto para revisão manual
                pil_rej  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                fname_rej = make_filename(pdf_stem, page_num + 1, idx, pil_rej)
                dest_rej  = os.path.join(DATASET_DIR, "rejeitadas", "filtro_texto", fname_rej)
                pil_rej.save(dest_rej)
                metadata_rows.append({
                    "pdf_origem": pdf_stem,
                    "pagina":     page_num + 1,
                    "origem":     "renderizada",
                    "caption":    "",
                    "categoria":  "filtro_texto",
                    "confianca":  0.0,
                    "largura":    crop_bgr.shape[1],
                    "altura":     crop_bgr.shape[0],
                    "path":       dest_rej,
                })
                continue

            # Converte para PIL RGB
            pil_img  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            w, h     = pil_img.size
            filename = make_filename(pdf_stem, page_num + 1, idx, pil_img)

            # ── 4. Converte bbox para pontos PDF e busca caption ─────────
            pdf_rect = pixels_to_pdf_rect(cx, cy, cw, ch, SCALE)
            caption  = find_caption(page, pdf_rect)

            row_base = {
                "pdf_origem": pdf_stem,
                "pagina":     page_num + 1,
                "origem":     "renderizada",
                "caption":    caption,
                "largura":    w,
                "altura":     h,
            }

            # ── Filtro: tamanho mínimo ───────────────────────────────────
            if w < MIN_SIZE or h < MIN_SIZE:
                dest = os.path.join(DATASET_DIR, "rejeitadas", "muito_pequena", filename)
                pil_img.save(dest)
                metadata_rows.append({**row_base,
                    "categoria": "rejeitada_tamanho", "confianca": 0.0, "path": dest})
                stats["rejeitada_tamanho"] += 1
                continue

            # ── Classificação CLIP ───────────────────────────────────────
            try:
                categoria, confianca = classify_image(pil_img)
            except Exception as e:
                print(f"  ⚠️  CLIP falhou em {filename}: {e}")
                stats["erros"] += 1
                continue

            # ── Filtro: confiança mínima ─────────────────────────────────
            if confianca < MIN_CONFIDENCE:
                dest = os.path.join(DATASET_DIR, "rejeitadas", "baixa_confianca", filename)
                pil_img.save(dest)
                metadata_rows.append({**row_base,
                    "categoria": f"baixa_confianca ({categoria})",
                    "confianca": round(confianca, 4), "path": dest})
                stats["rejeitada_confianca"] += 1
                continue

            # ── Salva na categoria ───────────────────────────────────────
            dest = os.path.join(DATASET_DIR, categoria, filename)
            pil_img.save(dest)
            metadata_rows.append({**row_base,
                "categoria": categoria,
                "confianca": round(confianca, 4), "path": dest})
            stats[categoria] += 1

    doc.close()

print("\n✅ Pipeline finalizado!")

# ==============================================================
# CÉLULA 11 — Salvar metadata.csv
# ==============================================================
csv_path   = os.path.join(DATASET_DIR, "metadata.csv")
fieldnames = ["pdf_origem", "pagina", "origem", "caption",
              "categoria",  "confianca", "largura", "altura", "path"]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"📄 metadata.csv salvo: {csv_path}")
print(f"   Total de registros: {len(metadata_rows)}")

# ==============================================================
# CÉLULA 12 — Relatório final
# ==============================================================
sep = "=" * 54
print(f"\n{sep}")
print("  📊 RELATÓRIO FINAL — Dataset de Imagens SB100")
print(sep)

total_aceitas = sum(stats[c] for c in CATEGORY_LABELS)
total_caption = sum(1 for r in metadata_rows if r.get("caption", ""))

print(f"\n✅ Imagens classificadas ({total_aceitas} total):")
for cat in CATEGORY_LABELS:
    bar = "█" * min(stats[cat], 40)
    print(f"   {cat:<20} {stats[cat]:>4}  {bar}")

print(f"\n📝 Com caption extraída:          {total_caption} / {len(metadata_rows)}")
print(f"\n❌ Rejeitadas — tamanho:           {stats['rejeitada_tamanho']}")
print(f"❌ Rejeitadas — baixa confiança:   {stats['rejeitada_confianca']}")
print(f"🚫 Páginas descartadas (só texto): {stats['paginas_sem_figura']}")
print(f"🚫 Crops descartados (só texto):   {stats['crop_texto']}")
print(f"⚠️  Erros:                           {stats['erros']}")
print(f"\n   Total geral: {len(metadata_rows)} imagens")
print(sep)

# ==============================================================
# FIM DO SCRIPT
# ==============================================================
#
# Por que renderizar em vez de extrair por xref?
# ─────────────────────────────────────────────
# Extração por xref retorna os objetos de imagem embutidos no PDF:
# logos de revista, watermarks, separadores decorativos, e fragmentos
# de figura (uma barra de gráfico pode ser um xref separado da legenda).
# O CLIP recebia esses fragmentos sem contexto e classificava mal.
#
# Renderizar a página completa e recortar via OpenCV garante que cada
# crop entregue ao CLIP é a figura como o leitor humano a vê — inteira,
# com eixos, legendas visuais e contexto espacial.
#
# Estrutura gerada:
#
# dataset_imagens/
# ├── grafico_linhas/
# ├── grafico_barras/
# ├── grafico_pizza/
# ├── fotografia/
# ├── diagrama/
# ├── fluxograma/
# ├── tabela_imagem/
# ├── mapa/
# ├── icone/
# ├── outro/
# ├── rejeitadas/
# │   ├── muito_pequena/
# │   └── baixa_confianca/
# └── metadata.csv
#
# metadata.csv — colunas:
#   pdf_origem | pagina | origem | caption | categoria |
#   confianca  | largura | altura | path
#
# caption: texto da legenda extraído da página PDF.
#   Ex: "Figura 3. Incidência de HLB por região em 2022."
#   Vazio para figuras sem legenda detectável.
#
# origem: sempre "renderizada" — todas as imagens vêm de
#   renderização de página, não de extração de xref.
#
# Para a equipe de ML:
#   from torchvision.datasets import ImageFolder
#   dataset = ImageFolder(root="dataset_imagens/")
