from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from duckduckgo_search import DDGS
from typing import List, Dict, Any

class InternetRAGSystem:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="mistral")
        self.llm = ChatOllama(model="mistral")
        self.vector_store = FAISS.from_texts([""], self.embeddings)
        
    def _web_search(self, query: str, num_results: int = 5) -> List[str]:
        """Get web content from search results"""
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
            return [f"Title: {res['title']}\nContent: {res['body']}" for res in results]

    def _modified_prompt(self):
        return ChatPromptTemplate.from_template(
            """Answer the question using web context and your knowledge:
            
            Web Context: {context}
            
            Question: {input}
            
            Format response as:
            [Web Context Answer]: ... 
            [General Knowledge]: ... 
            [Confidence]: High/Medium/Low"""
        )

    def ask(self, question: str) -> Dict[str, Any]:
        # Perform fresh web search for each question
        web_content = self._web_search(question)
        
        # Update vector store with new content
        self.vector_store = FAISS.from_texts(
            texts=web_content,
            embedding=self.embeddings
        )
        
        # Create processing chain
        retriever = self.vector_store.as_retriever()
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(
                self.llm,
                self._modified_prompt()
            )
        )
        
        response = chain.invoke({"input": question})
        return self._parse_response(response["answer"])

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the structured response"""
        breakdown = {
            "web_context": "",
            "general_knowledge": "",
            "confidence": ""
        }
        
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("[Web Context Answer]"):
                current_section = "web_context"
                breakdown[current_section] = line.split("]:")[1].strip()
            elif line.startswith("[General Knowledge]"):
                current_section = "general_knowledge"
                breakdown[current_section] = line.split("]:")[1].strip()
            elif line.startswith("[Confidence]"):
                breakdown["confidence"] = line.split("]:")[1].strip()
            elif current_section:
                breakdown[current_section] += " " + line
                
        return breakdown

# Usage
rag = InternetRAGSystem()
result = rag.ask('''What do you know about Chennai Mathematical Institute? For example Anant Mudgal was a student at CMI; Mathematics Educator | Part III Cambridge | 4x India IMO | Worldwide IMO Coach | Author of "The Book of Zero Knowledge" Now tell me who is Atul Anant?''')
print("\n=== Answer Breakdown ===")
print(f"Web Context: {result['web_context']}")
print(f"General Knowledge: {result['general_knowledge']}")
print(f"Confidence: {result['confidence']}")