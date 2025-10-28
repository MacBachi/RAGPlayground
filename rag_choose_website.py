import os
import sys
from typing import List, Dict, Any
# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
# OpenAI Imports
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    print("\033[91m❌ FATALER FEHLER: 'langchain-openai' Paket konnte nicht gefunden werden. Bitte installieren Sie es mit: pip install -U langchain-openai\033[0m")
    sys.exit(1)

# ANSI-Codes für Farben
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
ENDC = "\033[0m"

# --- KONFIGURATION UND PRÜFUNG ---

# Kritische Prüfung: OPENAI_API_KEY (Kurzfassung)
if not os.getenv("OPENAI_API_KEY"):
    print(f"\n{RED}❌ FEHLER: Die Umgebungsvariable 'OPENAI_API_KEY' ist nicht gesetzt.{ENDC}")
    sys.exit(1)

# URL-Prüfung: Muss als Parameter übergeben werden
if len(sys.argv) < 2:
    print(f"\n{RED}❌ FEHLER: Die Webseite fehlt!{ENDC}")
    print(f"{BLUE}Usage: python {sys.argv[0]} <WEBSEITEN_URL>{ENDC}")
    print(f"{BLUE}Example: python {sys.argv[0]} https://www.heise.de/security{ENDC}")
    sys.exit(1)

# Setze die übergebene URL
URL = sys.argv[1]

# Setze USER_AGENT
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "LangChainRAGScript/1.0"


# --- FUNKTIONEN FÜR ERROR-RESISTENZ ---

def load_documents_safe(url: str) -> List[Any]:
    """Lädt Dokumente und fängt Netzwerk- oder DNS-Fehler ab."""
    print(f"\n{BLUE}🚀 Starte: Laden der Dokumente von {url}...{ENDC}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        if not documents:
            raise ValueError("Dokumenten-Loader gab eine leere Liste zurück.")
        print(f"{GREEN}✅ Erfolg: {len(documents)} Dokumente geladen.{ENDC}")
        return documents
    except Exception as e:
        print(f"{RED}❌ FATALER FEHLER beim Laden der Dokumente von {url}: {e}{ENDC}")
        print(f"{RED}   Mögliche Ursachen: DNS-Auflösung, Netzwerkprobleme, URL falsch oder 'bs4' fehlt.{ENDC}")
        sys.exit(1)

def embed_and_store_safe(documents: List[Any]) -> Any:
    """Zerlegt, bettet ein und speichert die Vektoren, fängt API-Fehler ab."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        transformed_documents = text_splitter.split_documents(documents)
        print(f"{BLUE}✂️ Dokumente zerlegt in {len(transformed_documents)} Chunks.{ENDC}")

        print(f"{BLUE}🧠 Starte: Erstellung von Embeddings und FAISS Index...{ENDC}")
        embedding_model = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(transformed_documents, embedding_model)
        print(f"{GREEN}✨ Erfolg: Vektor-Store erstellt.{ENDC}")
        return vector_store
    except Exception as e:
        print(f"{RED}🔥 FATALER FEHLER bei der OpenAI API Interaktion oder FAISS: {e}{ENDC}")
        print(f"{RED}   Mögliche Ursachen: Quota (429), ungültiger API Key oder fehlende Berechtigungen.{ENDC}")
        sys.exit(1)


def run_rag_pipeline(vector_store: Any, question: str):
    """Definiert und führt die RAG-Pipeline aus."""
    try:
        retriever = vector_store.as_retriever()
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(temperature=0)
        output_parser = StrOutputParser()

        pipeline = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | model 
            | output_parser
        )

        print(f"{BLUE}\n❓ Starte: Abfrage der RAG-Pipeline mit Frage: '{question}'...{ENDC}")
        answer = pipeline.invoke(question)
        
        print(f"\n{GREEN}--- ANTWORT (ERFOLG) ---{ENDC}")
        print(answer)
        print(f"{GREEN}------------------------{ENDC}")

    except Exception as e:
        print(f"{RED}❌ FATALER FEHLER während der RAG-Pipeline-Ausführung: {e}{ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    
    # 1. Dokumente laden
    documents = load_documents_safe(URL)
    
    # 2. Embeddings erstellen
    vector_store = embed_and_store_safe(documents)
    
    # 3. RAG-Pipeline ausführen
    run_rag_pipeline(vector_store, "Was sind die Hauptthemen dieser Seite?")
