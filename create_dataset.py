import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

api_key = os.getenv("LANGSMITH_API_KEY")

if not api_key:
    raise ValueError("LANGSMITH_API_KEY not found in .env")

client = Client(api_key=api_key)

print("LangSmith API key loaded successfully!")
dataset_name = "Financial-RAG-Annual-Report"

# 14 manually verified test examples
evaluation_data = [
    {
        "question": "What was the total revenue in 2025?",
        "answer": "$219 million",
        "relevant_pages": [2]
    },
    {
        "question": "What was the platform revenue in 2025?",
        "answer": "$181 million",
        "relevant_pages": [2]
    },
    {
        "question": "What was the volume-based revenue in 2025?",
        "answer": "$38 million",
        "relevant_pages": [2]
    },
    {
        "question": "What was the recurring revenue in 2025?",
        "answer": "$205 million",
        "relevant_pages": [2]
    },
    {
        "question": "What was the GAAP gross margin in 2025?",
        "answer": "72%",
        "relevant_pages": [2]
    },
    {
        "question": "What was the GAAP operating margin in 2025?",
        "answer": "3%",
        "relevant_pages": [2]
    },
    {
        "question": "What was the GAAP diluted EPS in 2025?",
        "answer": "$(0.02) per share",
        "relevant_pages": [2]
    },
    {
        "question": "What was the year-end backlog in 2025?",
        "answer": "$254 million",
        "relevant_pages": [2]
    },
    {
        "question": "Who was the CEO of PDF Solutions?",
        "answer": "John K. Kibarian",
        "relevant_pages": [2, 17]
    },
    {
        "question": "How many employees did PDF Solutions have as of December 31, 2025?",
        "answer": "600 employees",
        "relevant_pages": [17]
    },
    {
        "question": "How many operating segments does PDF Solutions report?",
        "answer": "One operating segment",
        "relevant_pages": [78]
    },
    {
        "question": "How much variable-rate debt did the company have?",
        "answer": "$68.1 million",
        "relevant_pages": [47]
    },
    {
        "question": "What would be the annual interest expense impact of a 100-basis-point increase in interest rates?",
        "answer": "Annual interest expense would increase by approximately $0.4 million.",
        "relevant_pages": [47]
    },
    {
        "question": "Mention one technology-related risk faced by PDF Solutions.",
        "answer": "The company's research and development investments may fail to keep pace with technological changes or market needs.",
        "relevant_pages": [19]
    }
]

# Create dataset
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Ground-truth evaluation dataset for PDF Solutions 2025 Annual Report Financial RAG."
)

# Upload examples
client.create_examples(
    inputs=[
        {
            "question": item["question"],
            "relevant_pages": item["relevant_pages"]
        }
        for item in evaluation_data
    ],
    outputs=[
        {
            "answer": item["answer"]
        }
        for item in evaluation_data
    ],
    dataset_id=dataset.id
)

print("Dataset created successfully!")
print("Dataset name:", dataset_name)
print("Total test questions:", len(evaluation_data))
