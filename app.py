import os
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(
    page_title="CampusPilot AI",
    page_icon="✈️",
    layout="wide"
)

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# =========================================================
# 1. APPLICATION & AUTHENTICATION PROTOCOL
# =========================================================


st.title("✈️ CampusPilot AI — Student Resolution Hub")
st.caption("Automated Multi-Agent Assistance for University Administrative Workflows")

# Core Gate Security
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔐 Student Portal Secure Login")
    student_id = st.text_input("University Register Number / ID", value="2026VITSE042")
    password = st.text_input("Password", type="password", value="••••••••")
    
    if st.button("Authenticate"):
        if student_id and len(password) > 0:
            st.session_state.authenticated = True
            st.session_state.student_id = student_id
            st.rerun()
        else:
            st.error("Invalid Credentials Provided")
    st.stop()

# =========================================================
# 2. VECTOR BASE SEARCH TOOL (NATIVE CREWAI TOOL)
# =========================================================
@tool("Search College Policy Database")
def search_policy_database(query: str) -> str:
    """Searches the university knowledge base, SOPs, handling departments, 
    and document guidelines for college issues."""
    mock_db = {
        "id card": "Department: Student Affairs | Location: Block A, Room 102 | Cost: 200 INR paid at Accounts Block | Requirements: Copy of FIR/Undertaking form, passport photo | Processing Time: 3 working days | Timings: 9 AM - 2 PM.",
        "attendance": "Department: Academic Cell & Respective HOD | Policy: Medical Regularization requires official medical certificate signed by a civil surgeon + prescriptions, submitted within 5 working days of rejoining. OD (Official Duty) requires pre-approval from event coordinator. Resolution: 48 hours.",
        "hostel": "Department: Hostel Administration & Estate Office | Process: Raise a ticket in the student warden portal or log in the Block Register. Maintenance handles plumbing/electrical within 24 hours. Emergency shifts require Warden written sign-off.",
        "fee": "Department: Accounts & Finance Block | Policy: Late registration penalties accumulate at 100 INR/day. Online payment receipts take 24 hours to reconcile. Discrepancies require bank transaction statement copies.",
        "exam": "Department: Controller of Examinations (CoE) | Location: Central Exam Tower | Policy: Duplicate hall tickets issued up to 2 hours before exam with a fine of 500 INR. Exam registration changes locked 14 days prior to semester commencement."
    }
    
    normalized = query.lower()
    for key, context in mock_db.items():
        if key in normalized:
            return context
    return "Department: Central Administration Helpdesk | Location: Main Admin Block, Counter 1 | Process: Visit counter with student ID and a written statement. Expected Resolution: 3-5 business days."

# =========================================================
# 3. LAYOUT RENDERING & USER CONTROLS
# =========================================================
st.sidebar.markdown(f"**🟢 Authenticated Student**")
st.sidebar.text(f"ID: {st.session_state.student_id}")
if st.sidebar.button("Log Out"):
    st.session_state.authenticated = False
    st.rerun()

st.markdown("### Describe your campus issue below:")
student_issue = st.text_area(
    "What problem are you facing? (e.g., I lost my ID card, need attendance regularization, hostel room issues, etc.)",
    placeholder="Describe your issue in detail..."
)

# =========================================================
# 4. ORCHESTRATION PIPELINE
# =========================================================
if st.button("Launch CampusPilot System", type="primary"):
    if not student_issue.strip():
        st.warning("Please type a clear problem description first.")
    else:
        with st.spinner("🧠 CampusPilot Agents are collaborating over the university database..."):
            
            # Explicit model initialization mapping straight to the pipeline
            # UPDATE this block in app.py to match this clean LiteLLM native format:
            # UPDATE this block in app.py to force the LiteLLM routing pathway:
            
            llm_engine = LLM(
                model="gemini/gemini-2.5-flash",
               
                temperature=0.4
        )
            

            # Agent 1: Intent Analyzer
            intent_analyzer = Agent(
                role='Intent Analyzer',
                goal='Analyze the raw text student problem, extract exact category and department, and detect structural urgency.',
                backstory='You act as the primary routing desk for university workflows.',
                verbose=True,
                llm=llm_engine
            )

            # Agent 2: Knowledge Retrieval Agent (RAG Tool User)
            knowledge_retrieval = Agent(
                role='Knowledge Retrieval Agent',
                goal='Use your database lookup tool to fetch accurate campus policies and strict guidelines.',
                backstory='You are a database lookup unit. You do not manufacture responses without tool execution confirmation.',
                tools=[search_policy_database],  
                verbose=True,
                llm=llm_engine
            )

            # Agent 3: Resolution Planner
            resolution_planner = Agent(
                role='Resolution Planner',
                goal='Create a highly logical, simple step-by-step resolution plan covering timings, locations, and roadblocks.',
                backstory='You transform rigid administrative constraints into simple, supportive student guides.',
                verbose=True,
                llm=llm_engine
            )

            # Agent 4: Email & Communication Generator
            email_generator = Agent(
                role='Email Generator',
                goal='Draft an impeccable academic formal email application, a quick mobile text message, and a crisp executive summary.',
                backstory='You are a professional scribe. Your email outputs use clear brackets like [Your Name] for personalization.',
                verbose=True,
                llm=llm_engine
            )

            # Assign Tasks
            t1 = Task(
                description=f"Classify category, urgency, and targeted department for: '{student_issue}'",
                expected_output="A structured classification specifying Category, Urgency, and Department.",
                agent=intent_analyzer
            )
            t2 = Task(
                description="Query the policy database tool to pull strict rules, locations, point of contacts, and timelines.",
                expected_output="Direct policy documentation context text relating to the identified problem category.",
                agent=knowledge_retrieval
            )
            t3 = Task(
                description="Synthesize context into a clean, chronological student roadmap. Highlight office hours, document requirements, and common pitfalls.",
                expected_output="Markdown step-by-step student roadmap checklist.",
                agent=resolution_planner
            )
            t4 = Task(
                description="Generate a professional email draft to the department head, a brief WhatsApp update text, and a 2-sentence tactical summary.",
                expected_output="Markdown document with clearly marked sections: ### 📬 Official Email Draft, ### 💬 Instant Message Template, and ### 📝 Summary.",
                agent=email_generator
            )

            # Form Crew
            crew = Crew(
                agents=[intent_analyzer, knowledge_retrieval, resolution_planner, email_generator],
                tasks=[t1, t2, t3, t4],
                process=Process.sequential
            )
            
            # Execute workflow
            final_report = crew.kickoff()
            
            st.success("✅ Problem Analyzed and Resolution Package Compiled!")
            st.markdown("---")
            st.markdown(final_report)
