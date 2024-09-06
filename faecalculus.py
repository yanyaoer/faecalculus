#!/usr/bin/env python3

import argparse

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# refs:
# <https://python.langchain.com/v0.1/docs/use_cases/code_understanding/>

parser = argparse.ArgumentParser(
    prog="Codebase-QA",
    description="How does the program work",
)
parser.add_argument("-p", "--repo_path", default="/Users/yanyao/data/fork/rodio")
parser.add_argument("-l", "--language", default="rust")
parser.add_argument("-i", "--index_name")
parser.add_argument("-m", "--mode", default="qa")
parser.add_argument("--debug", default=False)
parser.add_argument("-q", "--question", default="how to run the main function")

config = parser.parse_args()
print(config, Language[config.language.upper()])

if config.debug:
    import langchain

    langchain.debug = True


class CodebaseQA(object):
    def __init__(self):
        llm_name = "codegemma"
        embeddings_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.repo = config.repo_path
        self.language = config.language
        self.cache = "./chroma_db"
        self.collection_name = config.index_name or f'idx_{self.repo.split("/")[-1]}'

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
        self.llm = ChatOllama(model=llm_name)

    def build_index(self):
        loader = GenericLoader.from_filesystem(
            self.repo,
            glob="**/*",
            suffixes=[".rs"],  # TODO generate by language
            parser=LanguageParser(language=config.language, parser_threshold=500),
        )
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=config.language, chunk_size=2000, chunk_overlap=200
        )
        texts = splitter.split_documents(documents)

        print(len(documents), len(texts))

        Chroma.from_documents(
            texts,
            self.embeddings,
            persist_directory=self.cache,
            collection_name=self.collection_name,
        )

    def ask(self, question="how it works?"):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )

        db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.cache,
            collection_name=self.collection_name,
        )

        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Respond in the following locale: zh-cn, Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        qa = create_retrieval_chain(retriever_chain, document_chain)
        result = qa.invoke({"input": question})
        print(question, prompt)
        print(result["answer"])

    def start(self):
        if config.mode == "index":
            self.build_index()
        else:
            self.ask(question=config.question)


if __name__ == "__main__":
    CodebaseQA().start()
