### **Assessment 1: AI Agent Workflow Automation**

**Objective:**
An AI-driven automation workflow that classifies incoming leads, extracts structured data (Name, Company, Intent), saves them to a database, and generates an automated response.

**Architecture:**

* **Orchestrator (The Brain):** n8n (running in Docker).
* **Intelligence:** Google gemini-3-flash-preview (via API).
* **Database Service:** Python FastAPI + SQLite (Stores results).
* **Containerization:** Docker Compose.

---

### **1. Setup & Installation**

**Prerequisites:**

* Docker & Docker Compose installed.
* Google Gemini API Key.

**Steps:**

1. **Clone/Open** the project folder `Assesment-1`.
2. **Start the System:**
```bash
docker compose up -d --build

```


3. **Access n8n:**
Open [http://localhost:5678](https://www.google.com/search?q=http://localhost:5678) in your browser.
4. **Import Workflow:**
* In n8n, go to menu -> **Import from File**.
* Select the `workflow.json` file included in this submission.
* **Important:** Open the "Gemini AI" node and paste your API Key in the URL field.


5. **Activate:**
Click **"Execute Workflow"** or **"Active"** toggle.

---

### **2. Prompt Strategy**

To ensure consistent and parsable outputs from the LLM, I utilized a **Strict Output Constrained** prompt strategy.

* **Role Definition:** Implicitly defined by asking it to "Analyze this lead."
* **Constraint Enforcement:** The prompt explicitly commands: *"Return ONLY raw JSON"* and *"Extract 'intent' (Sales, Support, Spam)..."*.
* **One-Shot Extraction:** Instead of multiple API calls, a single prompt extracts classification, confidence, entities (Name/Company), and generates the draft response simultaneously to reduce latency and cost.

**The Prompt Used:**

> "Analyze this lead. Extract 'intent' (Sales, Support, Spam), 'confidence' (0.0-1.0), and write a polite 'response'. Return ONLY raw JSON. Message: {{message}}"

---

### **3. Hallucination Reduction**

AI models can sometimes invent facts. I implemented the following guardrails to minimize this:

1. **Confidence Scoring:** The model is required to return a `confidence` score (0.0 - 1.0). In a production environment, logic is added to route low-confidence (< 0.7) leads to human review rather than sending an auto-reply.
2. **Restricted Classification:** The model is restricted to three specific intents: `Sales`, `Support`, or `Spam`. It cannot invent new categories.
3. **Source Grounding:** The prompt asks to analyze "this lead" specifically, discouraging the model from using outside knowledge to fabricate details about the user's company.

---

### **4. Error Handling**

Reliability is handled at both the Orchestration (n8n) and Service (Python) levels:

* **JSON Parsing (n8n):** LLMs often wrap JSON in markdown (e.g., ````json`). I implemented a specific **Code Node** in n8n to strip these characters before parsing, preventing syntax errors that would break the workflow.
* **Database Resilience (Python):** The FastAPI backend uses `try/except` blocks. If the database is locked or the data payload is malformed, it catches the error, logs it specifically (visible in Docker logs), and returns a 500 error to n8n, allowing the workflow to handle the failure gracefully.
* **Service Isolation:** Since the AI logic is decoupled from the Database service, a failure in the AI API (e.g., rate limit) does not crash the database, and vice versa.

---

### **5. Testing the Workflow**

You can simulate a lead using `curl`:

```bash
curl -X POST http://localhost:5678/webhook-test/lead \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, I am Alice from Wonderland Inc. I need a quote for 50 licenses."}'

```

**Expected Output:**

```json
{
  "status": "success",
  "lead_id": 1,
  "ai_reply": "Hi Alice, thank you for reaching out! We'd be happy to provide a quote for 50 licenses for Wonderland Inc. Could you please share..."
}
```