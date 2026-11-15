import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from groq import AsyncGroq
from api.main import query_rag_system
from dotenv import load_dotenv
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GROQ_CLIENT=AsyncGroq(api_key=GROQ_API_KEY)
# 1. Set your LLM judge API key (or configure it via environment variables)


def run_evaluation():
    # 2. Load your test dataset
    # In a real pipeline, you can dynamically generate 'answer' and 'contexts' 
    # by calling your RAG pipeline functions programmatically here.
    test_cases = [
        {"question": "What is the company policy on remote work?", "ground_truth": "Employees get a $500 stipend..."},
        {"question": "How many PTO days do we get?", "ground_truth": "Full-time staff get 20 days annually..."},
    ]
    generated_data=[]
    
    # 2. Dynamically run your RAG app against every test question
    print("Running RAG evaluation against metrics...")
    
    for case in test_cases:
        question=case["question"]
         # calling the actual backend code
        result=query_rag_system(question)
        
        generated_data.append({
            "question": question,
            "answer": result["answer"],          # What your LLM actually generated
            "contexts": result["contexts"],      # What your vector DB actually retrieved
            "ground_truth": case["ground_truth"] # The human-verified correct answer
        })
        
    # Convert generated data list to Hugging Face Dataset format required by RAGAS
    dataset = Dataset.from_list(generated_data)

    # 3. Score everything using RAGAS evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,  # Did retrieval fetch the right chunks?
            context_recall,     # Did retrieval miss any necessary info?
            faithfulness,       # Is the answer derived *only* from context (no hallucinations)?
            answer_relevance,   # Does the answer actually address the user's question?
        ]
    )

    # 4. Convert results to a pandas dataframe and display/save
    eval_df = result.to_pandas()
    print("\n--- Evaluation Results Summary ---")
    print(eval_df[["question", "context_precision", "faithfulness", "answer_relevance"]])
    
    # Save results to a CSV report 
    # eval_df.to_csv("eval/evaluation_report.csv", index=False)
    print("\nDetailed report saved to eval/evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()