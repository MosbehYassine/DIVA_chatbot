import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

try:
    import langchain_community
    print(f"langchain_community location: {langchain_community.__file__}")
except ImportError as e:
    print(f"Error importing langchain_community: {e}")

try:
    import langchain_community.vectorstores
    print(f"langchain_community.vectorstores imported successfully")
    print(f"dir(langchain_community.vectorstores): {dir(langchain_community.vectorstores)}")
except ImportError as e:
    print(f"Error importing langchain_community.vectorstores: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("FAISS imported successfully")
except ImportError as e:
    print(f"Error importing FAISS: {e}")
