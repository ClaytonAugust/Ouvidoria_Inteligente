# Ouvidoria Inteligente — passo a passo

## 1. Google Colab
1. Abra um novo notebook no Colab.
2. Faça upload destes arquivos para a mesma pasta:
   - `manifestacoes.json`
   - `analise_comparativa.ipynb`
   - `deteccao_duplicatas.ipynb`
   - `chunking_manifestacoes.ipynb`
3. Em uma célula, execute:
```bash
!pip install -r requirements.txt
```
4. Abra e execute os três notebooks célula por célula.
5. Baixe os notebooks executados.

## 2. Jupyter local
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## 3. Streamlit
Na pasta do projeto:
```bash
streamlit run ouvidoria.py
```
Também existe `app_ouvidoria.py`, conforme o nome da entrega do enunciado.

## 4. Observação sobre embeddings
Na primeira execução, `sentence-transformers` pode precisar baixar o modelo. Em ambiente sem internet, o modelo precisa estar previamente em cache ou ser substituído por um modelo local disponível.
