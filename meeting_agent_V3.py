import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os

# ── Streamlit app setup ──────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Meeting Agent 📝", layout="wide")
st.title("AI Meeting Preparation Agent 📝")

# ── Sidebar for API keys ─────────────────────────────────────────────────────────
st.sidebar.header("API Keys")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
serper_api_key = st.sidebar.text_input("Serper API Key", type="password")

# ── Main logic runs only when both keys are provided ─────────────────────────────
if anthropic_api_key and serper_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    claude = LLM(
        model="claude-3-5-sonnet-20240620",
        temperature=0.7,
        api_key=anthropic_api_key
    )
    search_tool = SerperDevTool()

    # ── User inputs ──────────────────────────────────────────────────────────────
    company_name     = st.text_input("Enter the company name:")
    meeting_objective = st.text_input("Enter the meeting objective:")
    attendees        = st.text_area("Enter the attendees and their roles (one per line):")
    meeting_duration = st.number_input(
        "Enter the meeting duration (in minutes):",
        min_value=15, max_value=180, value=60, step=15
    )
    focus_areas      = st.text_input("Enter any specific areas of focus or concerns:")

    # ── Agent definitions (expanded goals & backstories) ─────────────────────────
    context_analyzer = Agent(
        role="Corporate Diligence & Context Specialist",
        goal="""Deliver a forensic-level, 360-degree background dossier covering
        company structure, leadership bios, product portfolio, strategic milestones,
        patent/tech moat signals, financial KPIs, ESG posture, and current public
        sentiment—then translate every datapoint into concrete meeting relevance.""",
        backstory="""A former McKinsey & BCG consultant with 15 years in M&A due
        diligence for Fortune 100 clientele. You routinely unpack S-1 filings,
        10-Ks, global press releases, analyst calls, patent libraries, Glassdoor
        reviews, academic papers, and regulatory databases. Your trademark is
        relentless triangulation—no stone unturned, no unverified claim.""",
        verbose=True,
        allow_delegation=False,
        llm=claude,
        tools=[search_tool]
    )

    industry_insights_generator = Agent(
        role="Senior Industry & Competitive Intelligence Analyst",
        goal="""Surface macro-economic forces, value-chain dynamics, technology
        disruptors, funding flows, regulatory headwinds/tailwinds, white-space
        opportunities, and threat vectors—then map where the target company sits
        today and could pivot tomorrow.""",
        backstory="""Ex-Gartner VP of Research who authored multiple Magic
        Quadrant reports and briefed C-suites on $1 B+ market entries. Adept at
        mining primary research (S&P Capital IQ, Crunchbase, PitchBook),
        secondary literature (Harvard Business Review, Forrester, IDC), and
        synthesising them into strategic foresight frameworks.""",
        verbose=True,
        allow_delegation=False,
        llm=claude,
        tools=[search_tool]
    )

    strategy_formulator = Agent(
        role="Strategic Meeting Architect & Facilitator",
        goal="""Engineer a minute-by-minute agenda that maximises decision
        velocity, ensures stakeholder alignment, pre-emptively mitigates conflict,
        and produces a concrete post-meeting action map with owners, metrics, and
        deadlines.""",
        backstory="""Former PMO lead at a Big 4 consultancy; designed and ran more
        than 300 executive workshops for digital-transformation, JV kick-offs, and
        multi-billion procurement negotiations. Certified SCRUM Master and
        Prosci-ADKAR change practitioner—your sessions always end with measurable
        next steps.""",
        verbose=True,
        allow_delegation=False,
        llm=claude
    )

    executive_briefing_creator = Agent(
        role="C-Suite Communications & Storytelling Specialist",
        goal="""Transform dense analysis into a crystalline brief that a busy
        board member can absorb in < 2 minutes, highlighting strategic stakes,
        data-driven narratives, and visually-scannable takeaways.""",
        backstory="""A former speechwriter to two Fortune 50 CEOs and an award-
        winning financial journalist. You blend narrative craft with data
        precision, wielding tables, call-outs, and rhetoric so leadership grasps
        the ‘why’ and the ‘so-what’ instantly.""",
        verbose=True,
        allow_delegation=False,
        llm=claude
    )

    # ── Task definitions (unchanged but still exhaustive) ────────────────────────
    context_analysis_task = Task(
        description=f"""
        ***DO FIRST – Rapid Web Reconnaissance***  
        1. Run **≥ 3** Serper queries combining “{company_name}” with
           “earnings”, “partnership”, “acquisition”, and “{meeting_objective}”.  
        2. From the **top 5** results per query (≤ 12 months old) capture:  
           • Headline & publication date  
           • One-sentence takeaway  
           • Markdown hyperlink.

        ***DO SECOND – Corporate Deep-Dive***  
        – Explain {company_name}'s business model, flagship products/services,
          geographic revenue split, and latest **financial KPIs** (revenue, YoY
          growth, gross margin, EBITDA).  
        – Map the **3–4** closest competitors with clear differentiators
          (price, tech, distribution, brand).  
        – Extract any relevant ESG or regulatory issues.

        ***DO THIRD – Meeting Relevance***  
        Tie every insight back to:  
        • Objective: “{meeting_objective}”  
        • Attendees & roles:  
        {attendees}  
        • Focus areas / concerns: {focus_areas}

        ***DELIVERABLE*** – concise markdown report (≤ 450 words):

        # Context Snapshot  
        ## Breaking News  
        | Date | Headline | Why it Matters |  
        |------|----------|---------------|  

        ## Company Profile  
        – Bullet summary (products, KPIs, competitors, ESG flags).

        ## Implications for the Meeting  
        – 3–5 bullets linking findings to objective/focus areas.

        Use **bold** for critical figures; maintain clean markdown.
        """,
        agent=context_analyzer,
        expected_output="Detailed context analysis with clear relevance to meeting objective."
    )

    industry_analysis_task = Task(
        description=f"""
        Build upon the context analysis to craft a forward-looking **industry
        intelligence brief** for {company_name}'s sector.

        ***DATA GATHERING***  
        • Serper queries: “<industry> 2025 outlook”, “emerging trends”,
          “regulatory changes”, “market size 2024-2028”.  
        • Extract quantitative data (CAGR, TAM, funding) from Gartner, McKinsey,
          S&P, Crunchbase; cite via markdown links.

        ***ANALYSIS FRAMEWORK***  
        1. **Macro Trends** – 3-4 forces shaping the landscape.  
        2. **Competitive 2×2 Matrix** – Leader | Challenger | Niche | Emerging;
           indicate {company_name}'s position.  
        3. **Opportunities & Threats** – Rate impact probability & magnitude
           vs. “{meeting_objective}”.

        ***DELIVERABLE*** – markdown brief (≤ 500 words):

        # Industry Pulse 2025  
        ## Macro Trends  
        – …

        ## Competitive Landscape  
        – Table + bullets.

        ## Opportunity/Threat Radar  
        – Numbered list linking back to {company_name}.

        Spotlight insights relevant to the next **18 months**.
        """,
        agent=industry_insights_generator,
        expected_output="Comprehensive industry analysis aligned with meeting goals."
    )

    strategy_development_task = Task(
        description=f"""
        Design a **high-impact, minute-by-minute strategy** for the
        {meeting_duration}-minute session with {company_name}.

        ***INPUTS***  
        – Context summary  
        – Industry brief  
        – Objective: “{meeting_objective}”  
        – Focus areas: {focus_areas}

        ***AGENDA RULES***  
        • Max **4** agenda blocks + closing.  
        • Reserve **≥ 5 min** Q&A.  
        • Assign owner per block (match attendee roles).  
        • For each block state: purpose, data reference, desired outcome
          (decision, alignment, info).

        ***DELIVERABLE*** – markdown table:

        | Time | Topic | Owner | Talking Points | Desired Outcome |  
        |------|-------|-------|----------------|-----------------|

        End with **Facilitator Tips** (bullets) tackling focus areas.  
        Limit length to **≤ 350 words**.
        """,
        agent=strategy_formulator,
        expected_output="Time-boxed strategy & agenda with facilitator guidance."
    )

    executive_brief_task = Task(
        description=f"""
        Produce a **board-ready executive brief** that a leader can absorb in
        **< 2 minutes**.

        ***STRUCTURE (use exact H1/H2/H3)***

        # Executive Summary  
        • Objective: “{meeting_objective}”  
        • Meeting length: {meeting_duration} min  
        • Attendees & roles (concise list)  
        • **Top 3 strategic outcomes** (bold verbs)

        # Key Insights  
        ## Company Snapshot  
        – 3 bullets  
        ## Industry Pulse  
        – 3 bullets  
        ## Risks & Mitigations  
        | Risk | Likelihood | Mitigation |  
        |------|------------|-----------|

        # Talking Points & Data  
        – Numbered list, ≤ 25 words each, with one statistic/example
          (*italicised*) + hyperlink.

        # Anticipated Questions & Answers  
        | Question | 2-line Answer | Backup Link |  
        |----------|--------------|-------------|

        # Recommendations & Next Steps  
        – 3–5 action bullets with owner + target date

        ***FORMATTING RULES***  
        • Bold, italics, tables for scannability.  
        • Total length **≤ 600 words**.  
        • Every external fact has inline markdown link.

        Ensure direct alignment with focus areas: {focus_areas}.
        """,
        agent=executive_briefing_creator,
        expected_output="Concise, data-rich executive brief ready for C-suite."
    )

    # ── Build and run the crew ───────────────────────────────────────────────────
    meeting_prep_crew = Crew(
        agents=[
            context_analyzer,
            industry_insights_generator,
            strategy_formulator,
            executive_briefing_creator
        ],
        tasks=[
            context_analysis_task,
            industry_analysis_task,
            strategy_development_task,
            executive_brief_task
        ],
        verbose=True,
        process=Process.sequential
    )

    if st.button("Prepare Meeting"):
        with st.spinner("AI agents are preparing your meeting…"):
            result = meeting_prep_crew.kickoff()
        st.markdown(result)

    # ── Sidebar instructions ────────────────────────────────────────────────────
    st.sidebar.markdown("""
    ## How to use this app
    1. Enter your API keys in the sidebar.  
    2. Provide the requested meeting details.  
    3. Click **Prepare Meeting** to generate your tailored preparation package.

    The AI agents will jointly:
    - Analyse the meeting context and company background  
    - Provide industry insights and trend foresight  
    - Develop a data-driven meeting strategy and agenda  
    - Craft a board-quality executive brief

    *This may take a few minutes—thanks for your patience!*  
    """)
else:
    st.warning("Please enter both API keys in the sidebar before proceeding.")
