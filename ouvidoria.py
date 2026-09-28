import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import plotly.express as px

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

@st.cache_data
def carregar():
    with open("manifestacoes.json", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

@st.cache_resource
def carregar_modelo(nome):
    return SentenceTransformer(nome)

@st.cache_data
def gerar_embeddings(textos, nome):
    modelo = carregar_modelo(nome)
    return modelo.encode(list(textos), normalize_embeddings=True)

df = carregar()
modelos = {
    "MiniLM multilíngue": "paraphrase-multilingual-MiniLM-L12-v2",
    "MPNet multilíngue": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
st.sidebar.title("⚙️ Configuração")
modelo_label = st.sidebar.selectbox("Modelo de embedding", list(modelos))
top_k = st.sidebar.slider("Top-K", 1, 10, 5)
nome_modelo = modelos[modelo_label]
E = gerar_embeddings(df["texto"].tolist(), nome_modelo)

st.title("🏛️ Ouvidoria Inteligente")
st.caption("Triagem semântica de manifestações cidadãs")

tab1,tab2,tab3,tab4 = st.tabs(["🔍 Busca Semântica","📋 Base Completa","🌐 Espaço Vetorial","🧩 Chunking"])

with tab1:
    st.subheader("Busca semântica")
    consulta = st.text_area("Descreva o problema:", placeholder="Ex.: rua com muitos buracos e dificuldade para os veículos...")
    if consulta.strip():
        q = gerar_embeddings([consulta], nome_modelo)[0]
        scores = E @ q
        ordem = np.argsort(scores)[::-1][:top_k]
        for i in ordem:
            s=float(scores[i])
            emoji="🟢" if s>0.7 else ("🟡" if s>0.5 else "🔴")
            st.markdown(f"### {emoji} {df.iloc[i]['id']} — {s:.3f}")
            st.write(df.iloc[i]["texto"])
            st.caption(f"Categoria oficial: {df.iloc[i]['categoria_oficial']}")

with tab2:
    st.subheader("Base completa")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("Gerar matriz de similaridade"):
        S=cosine_similarity(E)
        matriz=pd.DataFrame(S,index=df["id"],columns=df["id"])
        st.dataframe(matriz.style.format("{:.3f}"),use_container_width=True)

with tab3:
    st.subheader("Espaço vetorial")
    metodo=st.radio("Redução dimensional",["PCA"],horizontal=True)
    xy=PCA(n_components=2,random_state=42).fit_transform(E)
    plot=pd.DataFrame({"x":xy[:,0],"y":xy[:,1],"id":df["id"],
                       "categoria":df["categoria_oficial"],"texto":df["texto"]})
    fig=px.scatter(plot,x="x",y="y",color="categoria",hover_name="id",
                   hover_data=["texto"],title="Manifestações no espaço semântico")
    st.plotly_chart(fig,use_container_width=True)
    st.info("A proximidade no espaço vetorial representa similaridade segundo o modelo, enquanto a cor representa a categoria oficial fornecida na base.")

with tab4:
    st.subheader("Chunking de manifestação longa")
    texto=st.text_area("Cole uma manifestação:", height=220)
    estrategia=st.selectbox("Estratégia",["A — 250/40","B — 450/80"])
    tamanho,overlap=(250,40) if estrategia.startswith("A") else (450,80)
    if texto.strip():
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=tamanho,chunk_overlap=overlap,
            separators=["\n\n","\n",". ","; ",", "," ",""]
        )
        chunks=splitter.split_text(texto)
        st.write(f"{len(chunks)} chunks gerados — chunk_size={tamanho}, overlap={overlap}")
        V=gerar_embeddings(chunks,nome_modelo)
        xy=PCA(n_components=2,random_state=42).fit_transform(V) if len(chunks)>=2 else np.zeros((1,2))
        for i,ch in enumerate(chunks):
            with st.expander(f"Chunk {i+1} — {len(ch)} caracteres"):
                st.write(ch)
        if len(chunks)>=2:
            cp=pd.DataFrame({"x":xy[:,0],"y":xy[:,1],"chunk":[f"Chunk {i+1}" for i in range(len(chunks))]})
            st.plotly_chart(px.scatter(cp,x="x",y="y",text="chunk",title="Chunks no espaço 2D"),use_container_width=True)
