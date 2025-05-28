###############################################################################
# ── 0.  RUNTIME PATCH: ensure SQLite ≥ 3.35.0 for Chroma / CrewAI  ────────────
###############################################################################
from packaging import version
import importlib, sys

try:
    import sqlite3
    if version.parse(sqlite3.sqlite_version) < version.parse("3.35.0"):
        # Swap in pysqlite3 (bundles SQLite 3.42+)
        pysqlite3 = importlib.import_module("pysqlite3")
        sys.modules["sqlite3"]  = pysqlite3
        sys.modules["_sqlite3"] = pysqlite3
except ModuleNotFoundError:
    raise RuntimeError(
        "Missing dependency `pysqlite3-binary`. "
        "Run:  pip install pysqlite3-binary  and retry."
    )

###############################################################################
# ── 1.  Imports  ──────────────────────────────────────────────────────────────
###############################################################################
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os, re

###############################################################################
# ── 2.  Streamlit UI  ─────────────────────────────────────────────────────────
###############################################################################
st.set_page_config(page_title="AI Meeting Agent 📝", layout="wide")
st.title("AI Meeting Preparation Agent 📝")

# Sidebar – API keys
st.sidebar.header("API Keys")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
serper_api_key    = st.sidebar.text_input("Serper API Key", type="password")

###############################################################################
# ── 3.  Main logic (only when keys provided)  ────────────────────────────────
###############################################################################
if anthropic_api_key and serper_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ["SERPER_API_KEY"]    = serper_api_key

    claude = LLM(
        model="claude-3-5-sonnet-20240620",
        temperature=0.7,
        api_key=anthropic_api_key
    )
    search_tool = SerperDevTool()

    # ── User inputs
    company_name      = st.text_input("Enter the company name:")
    meeting_objective = st.text_input("Enter the meeting objective:")
    attendees         = st.text_area("Enter the attendees and their roles (one per line):")
    meeting_duration  = st.number_input("Enter the meeting duration (in minutes):",
                                        min_value=15, max_value=180, value=60, step=15)
    focus_areas       = st.text_input("Enter any specific areas of focus or concerns:")

    ############################################################################
    # 4. Detect whether Castolin kiln-repair dataset is relevant
    ############################################################################
    cement_keywords = [
        r"\bcement\b", r"\bcement\s*plant\b", r"\bkiln\b", r"\brotary\s*kiln\b",
        r"\bcalciner\b", r"\bburner\b"
    ]
    query_blob      = f"{company_name} {meeting_objective} {focus_areas}".lower()
    sector_relevant = any(re.search(pat, query_blob) for pat in cement_keywords)

    ############################################################################
    # 5. Agent definitions
    ############################################################################
    # 5-A  Castolin retrieval agent (only added if relevant)
    if sector_relevant:
        castolin_retriever = Agent(
            role="Castolin Knowledge-Hub Retrieval Specialist",
            goal=("Surface kiln-repair application records, ROI analytics, success "
                  "stories, CRM feedback, and reference contacts."),
            backstory=("Data steward for Castolin Eutectic’s Central Knowledge Hub "
                       "covering 20 years of cement-plant wear-protection data."),
            verbose=True,
            allow_delegation=False,
            llm=claude
        )

    # 5-B  Your original agents
    context_analyzer = Agent(
        role="Corporate Diligence & Context Specialist",
        goal=("Deliver a forensic 360-degree dossier of the company and translate "
              "each finding into meeting relevance."),
        backstory=("Ex-McKinsey/BCG, 15 years M&A due-diligence, masters every "
                   "public source from SEC filings to patent libraries."),
        verbose=True, allow_delegation=False, llm=claude, tools=[search_tool]
    )

    industry_insights_generator = Agent(
        role="Senior Industry & Competitive Intelligence Analyst",
        goal=("Surface macro-economic forces, tech disruptors, regulatory shifts, "
              "and competitive moves; position the target company."),
        backstory=("Ex-Gartner VP Research; author of multiple Magic Quadrants."),
        verbose=True, allow_delegation=False, llm=claude, tools=[search_tool]
    )

    strategy_formulator = Agent(
        role="Strategic Meeting Architect & Facilitator",
        goal=("Engineer a minute-by-minute agenda that maximises decision "
              "velocity and produces measurable next steps."),
        backstory=("Former Big-4 PMO director; 300+ C-suite workshops delivered."),
        verbose=True, allow_delegation=False, llm=claude
    )

    executive_briefing_creator = Agent(
        role="C-Suite Communications & Storytelling Specialist",
        goal=("Transform dense analysis into a crystalline brief that a board can "
              "absorb in < 2 minutes."),
        backstory=("Ex-Fortune-50 CEO speechwriter; award-winning journalist."),
        verbose=True, allow_delegation=False, llm=claude
    )

    # Assemble agent list
    agents = [context_analyzer, industry_insights_generator,
              strategy_formulator, executive_briefing_creator]
    if sector_relevant:
        agents.insert(0, castolin_retriever)

    ############################################################################
    # 6. Task definitions
    ############################################################################
    tasks = []

    # 6-A  Castolin synthetic-data task (only if relevant)
    if sector_relevant:
        castolin_data_task = Task(
            description="""
            You have direct SQL/API access to **Castolin Central Knowledge Hub**.

            **If this meeting is *not* about cement-plant kiln maintenance, reply
            exactly with:**  
            *“No Castolin internal data relevant to the requested sector.”*

            **Otherwise** load the embedded YAML dataset (see below) and deliver:
            1. ROI Summary Table (include annualised € savings).  
            2. Three 80-word success stories with direct quotes.  
            3. Reference directory (mask emails: first initial + "***").

            Total output ≤ 600 words in the specified markdown structure.

            ```yaml
            applications_records:
              - id: APR-2023-001
                plant: Holcim Lägerdorf Plant
                country: Germany
                equipment: Rotary Kiln (Outlet section)
                product_used: CastoDur Diamond Plate 4666
                year: 2023
                downtime_reduction_hours: 68
                cost_saving_eur: 235000
                roi_months: 4.2
              - id: APR-2024-004
                plant: Heidelberg Materials Leimen
                country: Germany
                equipment: Rotary Kiln (Tire & Shell)
                product_used: CastoWear 3001 Weld Overlay
                year: 2024
                downtime_reduction_hours: 102
                cost_saving_eur: 412000
                roi_months: 3.5
              - id: APR-2024-012
                plant: CEMEX Broceni
                country: Latvia
                equipment: Calciner Knee
                product_used: Castolin Xuper 6256 Wire
                year: 2024
                downtime_reduction_hours: 57
                cost_saving_eur: 178000
                roi_months: 5.1
              - id: APR-2025-003
                plant: Buzzi Unicem Vernasca
                country: Italy
                equipment: Rotary Kiln (Inlet)
                product_used: EnDOtec® DO*327 Deposit
                year: 2025
                downtime_reduction_hours: 89
                cost_saving_eur: 305000
                roi_months: 3.9

            success_stories:
              - story_id: SS-2023-09
                plant: Holcim Lägerdorf
                quote: "Castolin's Diamond Plate halved our maintenance windows—game changer!"
                outcome: "Extended kiln life by 18 months, saved €235 k first year."
              - story_id: SS-2024-02
                plant: Heidelberg Materials Leimen
                quote: "ROI in under 4 months, unprecedented reliability."
                outcome: "Zero unplanned stoppages for 11 months."
              - story_id: SS-2025-01
                plant: Buzzi Unicem Vernasca
                quote: "Castolin overlay reduced shell warpage, exceeded OEM specs."
                outcome: "Projected €1.2 M savings over 5-year horizon."

            crm_references:
              - plant: CEMEX Broceni
                contact: Janis Ozols
                role: Maintenance Manager
                email: janis.ozols@cemex.com
                satisfaction_score: 9.7
              - plant: Holcim Lägerdorf
                contact: Claudia Meyer
                role: Production Engineer
                email: claudia.meyer@holcim.com
                satisfaction_score: 9.5

            directory_contacts:
              - plant: Holcim Lägerdorf
                person: Dr. Markus Hahn
                position: Plant Director
                email: markus.hahn@holcim.com
                phone: +49 451 789 123
              - plant: Heidelberg Materials Leimen
                person: Susanne Krüger
                position: Maintenance Lead
                email: s.krueger@heidelbergmaterials.com
                phone: +49 6224 567 890
              - plant: CEMEX Broceni
                person: Janis Ozols
                position: Maintenance Manager
                email: janis.ozols@cemex.com
                phone: +371 637 445 21
              - plant: Buzzi Unicem Vernasca
                person: Luca Bianchi
                position: Engineering Chief
                email: l.bianchi@buzziunicem.it
                phone: +39 0523 764 998
              - plant: Titan Cement Thessaloniki
                person: Maria Papadopoulou
                position: Reliability Engineer
                email: mpapad@titan.gr
                phone: +30 2310 998 332
            ```
            """,
            agent=castolin_retriever,
            expected_output=("Markdown with ROI table, success stories and "
                             "masked directory *or* the single-line notice "
                             "when sector not relevant.")
        )
        tasks.append(castolin_data_task)

    # 6-B  Public-source context analysis
    internal_clause = (
        "Integrate public sources with Castolin ROI data."
        if sector_relevant else
        "No Castolin data available; rely solely on public sources."
    )
    context_analysis_task = Task(
        description=f"""
        {internal_clause}

        1 · Rapid Web Recon – ≥ 3 Serper searches combining “{company_name}”
        with “earnings”, “partnership”, “acquisition”, “{meeting_objective}”.  
        2 · Company Deep-Dive – business model, KPIs, competitors, ESG flags.  
        3 · Meeting Relevance – link findings to objective, attendees, focus areas.

        Deliver ≤ 450 words markdown with headings:
        # Context Snapshot → ## Breaking News → ## Company Profile
        → ## Implications for the Meeting
        """,
        agent=context_analyzer,
        expected_output="Context analysis (public + optional internal)."
    )
    tasks.append(context_analysis_task)

    # 6-C  Industry analysis
    industry_analysis_task = Task(
        description=f"""
        Craft a ≤ 500-word industry intelligence brief.

        • Data: “<industry> 2025 outlook”, “maintenance trends”, etc.  
        • 3-4 Macro Trends; 2×2 Competitive Matrix incl. {company_name}.  
        • Opportunity/Threat radar (probability & magnitude).
        """,
        agent=industry_insights_generator,
        expected_output="Industry brief aligned with meeting goals."
    )
    tasks.append(industry_analysis_task)

    # 6-D  Strategy & agenda
    roi_note = ("Include Castolin ROI stories as social proof."
                if sector_relevant else
                "Do not reference Castolin internal data.")
    strategy_development_task = Task(
        description=f"""
        Minute-by-minute agenda for {meeting_duration}-minute meeting.

        • ≤ 4 blocks + closing; ≥ 5 min Q&A.  
        • Table: Time | Topic | Owner | Talking Points | Desired Outcome.  
        {roi_note}  
        Add facilitator tips (≤ 350 words).
        """,
        agent=strategy_formulator,
        expected_output="Agenda table + facilitator tips."
    )
    tasks.append(strategy_development_task)

    # 6-E  Executive brief
    roi_clause = ("Cite ≥ 2 Castolin ROI stats."
                  if sector_relevant else
                  "Exclude Castolin internal data.")
    executive_brief_task = Task(
        description=f"""
        Board-ready brief ≤ 600 words:

        • Executive Summary (objective, duration, attendees, 3 bold outcomes)  
        • Key Insights (company, industry{', Castolin ROI tie-ins' if sector_relevant else ''})  
        • Risks & Mitigations table  
        • Talking Points & Data (link every stat)  
        • Anticipated Q&A table  
        • Recommendations & Next Steps (owner + date)

        {roi_clause}
        """,
        agent=executive_briefing_creator,
        expected_output="Concise executive brief."
    )
    tasks.append(executive_brief_task)

    ############################################################################
    # 7. Build and run the Crew
    ############################################################################
    meeting_prep_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )

    if st.button("Prepare Meeting"):
        with st.spinner("AI agents are preparing your meeting…"):
            result = meeting_prep_crew.kickoff()
        st.markdown(result)

    # Sidebar help
    st.sidebar.markdown("""
    ## How to use this app
    1. Enter your API keys.  
    2. Fill in the meeting details.  
    3. Click **Prepare Meeting**.

    *If the sector involves cement-plant kiln repair, proprietary Castolin data
    will be included automatically; otherwise, analysis is public-source only.*
    """)
else:
    st.warning("Please enter both API keys in the sidebar before proceeding.")
