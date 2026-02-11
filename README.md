Focus on clean implementation, system design, and clear explanations rather than UI polish. Choose own preferred tech stack unless stated otherwise. 

Assessment 1 

**AI Agent Workflow Automation (n8n + LLM)** 

Objective 

Design and implement an AI-driven automation workflow for handling incoming leads, integrated with a simple web interface or API endpoint. 

Task 

Build a workflow that: 

* Accepts a lead message via: 


* Webhook, OR 


* Simple web form / API endpoint 




* Uses an LLM to: 


* Classify intent (Sales / Support / Spam) 


* Extract key fields (name, company, requirement) Stores structured data in a database or mock datastore 




* Sends an AI-generated automated response 


* Includes error handling and retry or fallback logic 



Requirements 

* Use n8n (preferred) or an equivalent workflow orchestration tool. 


* LLM via API or local model 


* Include a basic backend or web endpoint (Node.js, Python, FastAPI, Express, etc.) 



Must Explain 

* Prompt strategy 


* How hallucinations are reduced 


* Error-handling approach 



Submission 

* n8n workflow export (JSON). 


* Short README (1-2 pages). 


* Demo video 



---

Assessment 2 

**Mysoft Heaven (BD) Ltd.-Specific AI Chatbot (RAG) with Web Interface** 

Objective 

Create a web-based AI chatbot that answers questions strictly based on Mysoft Heaven (BD) Ltd.-provided documents and company information. 

Task 

Using the company profile and documents provided by Mysoft Heaven (BD) Ltd. (e.g., company overview, services, products, certifications, platforms, or project descriptions): 

* Build a Retrieval-Augmented Generation (RAG) pipeline 


* Use a vector database (FAISS, Pinecone, Milvus, or Weaviate) 


* Develop a simple web UI or API for chatbot interaction 


* Ensure the chatbot: 


* Responds only using the provided Mysoft Heaven (BD) Ltd. data 


* Does not generate answers outside the supplied information 




* Safely handle or reject unrelated or out-of-scope questions 



Must Explain 

* Document chunking strategy 


* Embedding model choice 


* How irrelevant or unsupported queries are handled 


* How the same architecture could support multiple companies in the future 



Bonus 

* Conversation memory 


* Confidence-based responses or fallback messaging 



---

Assessment 3 

**AI Agent System Design for Web-Based SaaS Platform** 

Objective 

Demonstrate system architecture and product-level thinking for an AI-powered web application. 

Task 

Design a system where companies can use a web-based platform to: 

* Connect social media pages 


* Receive AI-powered automated replies 


* Auto-tag leads (hot/warm/cold) 


* Sync data with a CRM system 



Deliverables 

* System architecture diagram 


* Explanation covering: 


* AI agents and services involved 


* Frontend-backend-AI data flow 


* Authentication and authorization 


* Data security and privacy 


* Cost optimization strategies 


* Failure scenarios and recovery handling 