import os
import sys
import argparse
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_verbose
from langchain_core.documents import Document

# OpenAI Imports
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    print("❌ FEHLER: 'langchain-openai' fehlt. Installieren Sie es mit: pip install -U langchain-openai")
    sys.exit(1)

# --- KONSTANTEN ---
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
ENDC = "\033[0m"

# --- HELFER-FUNKTIONEN (Unverändert) ---

def check_environment(verbose: bool):
    """Prüft auf OPENAI_API_KEY und aktiviert den Verbose Mode."""
    if verbose:
        try:
            set_verbose(True)
            print(f"{BLUE}🔍 Debugging: LangChain Verbose Mode aktiviert.{ENDC}")
        except Exception:
            print(f"{RED}❌ WARNUNG: set_verbose konnte nicht aktiviert werden.{ENDC}")

    if not os.getenv("OPENAI_API_KEY"):
        print(f"\n{RED}❌ FEHLER: Die Umgebungsvariable 'OPENAI_API_KEY' ist nicht gesetzt.{ENDC}")
        sys.exit(1)

def parse_arguments():
    """Definiert und parst Kommandozeilenargumente (nur file_path und -v)."""
    parser = argparse.ArgumentParser(
        description="Extrahiert Indicators of Compromise (IOCs) aus einem PDF-Dokument.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("file_path", type=str, help="Pfad zum zu analysierenden PDF-Dokument.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Aktiviert den LangChain Debugging-Modus.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"\n{RED}❌ FEHLER: Datei nicht gefunden: {args.file_path}{ENDC}")
        sys.exit(1)
        
    if not args.file_path.lower().endswith('.pdf'):
        print(f"\n{RED}❌ FEHLER: Nur PDF-Dateien werden unterstützt: {args.file_path}{ENDC}")
        sys.exit(1)
        
    return args

def load_documents_safe(file_path: str) -> List[Document]:
    """Lädt Dokumente nur mit PyPDFLoader."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except ImportError:
        print(f"{RED}❌ FATALER FEHLER: Das Paket 'pypdf' fehlt. Installieren Sie es mit: pip install pypdf{ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}❌ FATALER FEHLER beim Laden der Dokumente: {e}{ENDC}")
        sys.exit(1)

def embed_and_store_safe(documents: List[Document]) -> Any:
    """Zerlegt, bettet ein und speichert die Vektoren."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
        transformed_documents = text_splitter.split_documents(documents)
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 
        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_documents(transformed_documents, embedding_model)
        return vector_store
    except Exception as e:
        print(f"{RED}🔥 FATALER FEHLER bei Embedding/FAISS: {e}{ENDC}")
        sys.exit(1)

def format_docs(docs: List[Document]) -> str:
    """Wandelt eine Liste von Documents in einen einzigen String um."""
    return "\n\n".join([doc.page_content for doc in docs])

# --- FUNKTION MIT LCEL-KETTE ---

def run_simple_ioc_rag(vector_store: Any) -> str:
    """Definiert und führt die RAG-Pipeline zur einfachen IOC-Extraktion aus."""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    retriever = vector_store.as_retriever(k=10) 
    
    template = """
    Sie sind ein Security-Analyst. Ihre Aufgabe ist es, alle Indicators of Compromise (IOCs) aus dem bereitgestellten KONTEXT zu extrahieren.
    
    Geben Sie die IOCs als einfache, durch Zeilenumbrüche getrennte Liste aus.
    Fügen Sie KEINEN zusätzlichen Text, Erklärungen oder Überschriften hinzu.
    
    KONTEXT:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # KORREKTUR: Robuste, mehrstufige LCEL-Kette
    # 1. Startet mit RunnablePassthrough (erhält {})
    # 2. Führt Retrieval und Formatierung aus und weist das Ergebnis dem 'context' zu
    # 3. Das Prompt erhält nun das korrekte Dict {"context": String}
    
    pipeline = (
        RunnablePassthrough() # Startet die Kette und übergibt {}
        | RunnablePassthrough.assign(
            context=retriever | RunnableLambda(format_docs) # context ist ein String
        )
        # Jetzt hat der Input die Form: {"context": String}
        | prompt
        | model
    )

    print(f"{BLUE}\n❓ Starte: IOC-Extraktion der RAG-Pipeline...{ENDC}")
    try:
        # Die Kette wird aufgerufen und die leere Eingabe an RunnablePassthrough gesendet.
        analysis_result = pipeline.invoke({})
        return analysis_result.content
    except Exception as e:
        print(f"{RED}❌ FATALER FEHLER während der RAG-Pipeline-Ausführung: {e}{ENDC}")
        print(f"{RED}HINWEIS: Der 'dict'-Fehler liegt tief in der LangChain-Abhängigkeit. Bitte aktualisieren Sie alle Pakete!{ENDC}")
        sys.exit(1)

# --- MAIN FUNKTION ---

if __name__ == "__main__":
    
    # 0. Konfiguration & Argumente
    args = parse_arguments()
    check_environment(args.verbose)
    
    # 1. Dokumente laden
    print(f"{BLUE}🚀 Starte: Laden des Dokuments von {args.file_path}...{ENDC}")
    documents = load_documents_safe(args.file_path)
    print(f"{GREEN}✅ Erfolg: {len(documents)} Seiten geladen.{ENDC}")
    
    # 2. Embeddings erstellen und Vektor-Store initialisieren
    print(f"{BLUE}🧠 Starte: Erstellung von Embeddings und FAISS Index...{ENDC}")
    vector_store = embed_and_store_safe(documents)
    print(f"{GREEN}✨ Erfolg: Vektor-Store erstellt.{ENDC}")
    
    # 3. RAG-Pipeline zur IOC-Extraktion ausführen (mit LCEL)
    ioc_list_text = run_simple_ioc_rag(vector_store)

    # 4. Ausgabe
    print(f"\n{GREEN}--- EXTRAHIERTE IOCs (REINER TEXT) ---{ENDC}")
    print(ioc_list_text)
    print(f"{GREEN}-------------------------------------------{ENDC}")
