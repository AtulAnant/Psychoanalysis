import json
import time
from typing import List, Dict, Any

from duckduckgo_search import DDGS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


class InternetRAGSystem:
    def __init__(self):
        # 1) Embeddings + LLM both point at ollama://mistral
        self.embeddings    = OllamaEmbeddings(model="mistral")
        self.llm           = ChatOllama(model="mistral")
        # 2) We'll only build FAISS once we have snippets
        self.vector_store  = None

        self.history       = []
        self.response_cache: Dict[str, Dict[str, Any]] = {}

    def _web_search(self, query: str, num_results: int = 5) -> List[str]:
        """
        Use DuckDuckGo (via duckduckgo_search.DDGS) to get top‐`num_results` snippets for `query`.
        Returns a list of snippet strings. If no results, returns [].
        """
        try:
            snippets: List[str] = []
            # DDGS().text(...) yields dicts with keys 'title' and 'body'
            with DDGS() as ddgs:
                for hit in ddgs.text(query, max_results=num_results):
                    title = hit.get("title", "").strip()
                    body  = hit.get("body", "").strip()
                    if title and body:
                        snippets.append(f"Title: {title}\nContent: {body}")
                    elif body:
                        snippets.append(body)

            return snippets
        except Exception:
            # If DuckDuckGo scraping fails (network, parsing, etc.), return empty so LLM fallback triggers
            return []

    def _modified_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """Answer the question using web context and your knowledge:

            Web Context:
            {context}

            Question:
            {input}

            Format response as:
            [Web Context Answer]: …
            [General Knowledge]: …
            [Confidence]: High/Medium/Low
            """
        )

    def ask(self, question: str, rating: Any) -> Dict[str, Any]:
        # 1) Check cache
        if question in self.response_cache:
            return self.response_cache[question]

        # 2) Grab web snippets
        snippets = self._web_search(question)

        # 3) If no snippets, fall back to pure LLM‐only
        if not snippets:
            print("⚠️  No search results – falling back to LLM-only.")
            prompt = (
                f"Question: {question}\n\n"
                "Please answer to the best of your ability and mark [Confidence]."
            )
            # Pass a plain string to call_as_llm
            raw = self.llm.call_as_llm(prompt)
            return {
                "web_context": "",
                "general_knowledge": raw,
                "confidence": "Medium",
                "rating": rating
            }

        # 4) Build a FAISS index on those snippets
        self.vector_store = FAISS.from_texts(texts=snippets, embedding=self.embeddings)

        # 5) Retrieval + “stuff” chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(
                self.llm,
                self._modified_prompt()
            )
        )

        # 6) Invoke the chain with the question
        out = chain.invoke({"input": question})
        parsed = self._parse_response(out["answer"])
        parsed["rating"] = rating

        # 7) Cache & return
        self.response_cache[question] = parsed
        return parsed

    def _parse_response(self, text: str) -> Dict[str, str]:
        bd = {"web_context": "", "general_knowledge": "", "confidence": ""}
        section = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("[Web Context Answer]"):
                section = "web_context"
                bd[section] = line.split("]:", 1)[1].strip()
            elif line.startswith("[General Knowledge]"):
                section = "general_knowledge"
                bd[section] = line.split("]:", 1)[1].strip()
            elif line.startswith("[Confidence]"):
                bd["confidence"] = line.split("]:", 1)[1].strip()
            elif section:
                bd[section] += " " + line
        return bd

    def run_questionnaire(self, questions: Dict[str, str], answers: Dict[str, Any]) -> List[Dict[str, Any]]:
        for qid, question in questions.items():
            score = answers.get(qid, None)
            print(f"\n[Q{qid}] {question} (Predef Score: {score})")
            resp = self.ask(question, score)

            print("→ WebCtx:", resp["web_context"][:100], "…")
            print("→ GenKnow:", resp["general_knowledge"][:100], "…")
            print("→ Confidence:", resp["confidence"])
            print("→ Assigned Score:", resp["rating"])

            self.history.append({
                "question_id": qid,
                "question": question,
                "response": resp,
                "score": score,
            })
            time.sleep(2)

        return self.history

    def save_history(self, path="questionnaire_results.json"):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    base = r"D:\projects-2025\Psychoanalysis\Data"
    with open(base + r"\tree.json")    as f: questions = json.load(f)["questions"]
    with open(base + r"\answers.json") as f: answers   = json.load(f)["Answers"]

    rag = InternetRAGSystem()
    rag.run_questionnaire(questions, answers)
    rag.save_history()
    print("\n✅ Saved results to questionnaire_results.json")
    
    # Enquire about the defense mechanism based on input scores
    print(rag.ask(f"Based on these scores: {answers}, what defense mechanism is this person likely using?", None)["general_knowledge"])
