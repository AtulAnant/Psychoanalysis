import json
import time
import requests
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# === CONFIG (or move these into env vars) ===
GOOGLE_API_KEY    = "AIzaSyDB6u-7-8gBIWPBQ3A6Zbo8lIceE8oJayk"
SEARCH_ENGINE_ID  = "a11b657c12fd94032"


class InternetRAGSystem:
    def __init__(self):
        # 1) Embedding + LLM both point at ollama://mistral
        self.embeddings    = OllamaEmbeddings(model="mistral")
        self.llm           = ChatOllama(model="mistral")
        # 2) We’ll only build this once we have content
        self.vector_store  = None

        self.history       = []
        self.response_cache: Dict[str, Dict[str, Any]] = {}

    def _web_search(self, query: str, num_results: int = 5) -> List[str]:
        """Hit Google CSE and return a list of ‘Title\nContent’ chunks."""
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": query,
                "num": num_results,
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        out = []
        for item in data.get("items", []):
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            out.append(f"Title: {title}\nContent: {snippet}")
        return out

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
        # cache hit
        if question in self.response_cache:
            return self.response_cache[question]

        # 1) grab web snippets
        snippets = self._web_search(question)

        # 2) if we got nothing, fall back to pure LLM
        if not snippets:
            print("⚠️  No search results – falling back to LLM-only.")
            prompt = (
                f"Question: {question}\n\n"
                "Please answer to the best of your ability and mark [Confidence]."
            )
            # ── Only this line changed: pass `prompt` (a str), not a list of dicts :contentReference[oaicite:0]{index=0}
            raw = self.llm.call_as_llm(prompt)
            return {"web_context": "", "general_knowledge": raw, "confidence": "Medium", "rating": rating}

        # 3) build FAISS index
        self.vector_store = FAISS.from_texts(texts=snippets, embedding=self.embeddings)

        # 4) retrieval + stuff chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(
                self.llm,
                self._modified_prompt()
            )
        )

        # 5) invoke
        out = chain.invoke({"input": question})
        parsed = self._parse_response(out["answer"])
        parsed["rating"] = rating

        # cache & return
        self.response_cache[question] = parsed
        return parsed

    def _parse_response(self, text: str) -> Dict[str, str]:
        bd = {"web_context":"", "general_knowledge":"", "confidence":""}
        section = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("[Web Context Answer]"):
                section = "web_context"
                bd[section] = line.split("]:",1)[1].strip()
            elif line.startswith("[General Knowledge]"):
                section = "general_knowledge"
                bd[section] = line.split("]:",1)[1].strip()
            elif line.startswith("[Confidence]"):
                bd["confidence"] = line.split("]:",1)[1].strip()
            elif section:
                bd[section] += " " + line
        return bd

    def run_questionnaire(self, questions: Dict[str,str], answers: Dict[str,Any]) -> List[Dict[str,Any]]:
        est_secs = len(questions) * 4
        print(f"\nEstimated time: ~{est_secs//60} min {est_secs%60} sec\n")
        plt.barh(["Remaining"], [est_secs])
        plt.xlabel("Seconds"); plt.title("Estimated Wait")
        plt.xlim(0, max(60, est_secs+5))
        plt.show()

        for qid, question in questions.items():
            score = answers.get(qid) or None
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
