from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

class JobAnalysis(BaseModel):
    title: str
    company: str
    required_skills: list[str]
    preferred_skills: list[str]
    yoe_range: str
    salary_range: str
    remote_policy: str
    match_percentage: float
    missing_skills: list[str] 
 

def compare_skills_zero_shot(job_description: str, job_description2: str, job_description3: str, resume: str) -> JobAnalysis:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract job posting data. For title, use the exact job title only — no extra words. missing_skills should ONLY contain items from required_skills and preferred_skills that the candidate lacks. Do not invent new skills."},
            {"role": "user", "content": f"Job Description:\n{job_description}\n\nJob Description 2:\n{job_description2}\n\nJob Description 3:\n{job_description3}\n\nResume:\n{resume}"}
        ],
        response_format=JobAnalysis,
        temperature=0
    )
    return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens

def compare_skills_few_shot(job_description: str, job_description2: str, job_description3: str, resume: str) -> JobAnalysis:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract job posting data. For title, use the exact job title only — no extra words."
            " missing_skills should ONLY contain items from required_skills and preferred_skills that the candidate lacks. Do not invent new skills."
            "here is an example of the expected output format for a different job description and resume:"
            "\n\nJob Description:\nData Scientist role at a tech company. Required skills: Python, SQL, Machine Learning. Preferred skills: Deep Learning, Cloud Computing. YOE: 3-5 years. Salary: $100k-$150k. Remote policy: Hybrid.\n\nResume:\nCandidate has experience with Python and SQL, but not Machine Learning or the preferred skills.\n\nExpected Output:\n{\n  title: 'Data Scientist',\n  company: 'Tech Company',\n  required_skills: ['Python', 'SQL', 'Machine Learning'],\n  preferred_skills: ['Deep Learning', 'Cloud Computing'],\n  yoe_range: '3-5 years',\n  salary_range: '$100k-$150k',\n  remote_policy: 'Hybrid',\n  match_percentage: 50.0,\n  missing_skills: ['Machine Learning', 'Deep Learning', 'Cloud Computing']\n}"
            "second example:\n\nJob Description:\nSoftware Engineer role at a startup. Required skills: Java, Git, Agile. Preferred skills: Docker, Kubernetes. YOE: 2-4 years. Salary: $80k-$120k. Remote policy: Remote.\n\nResume:\nCandidate has experience with Java and Git, but not Agile or the preferred skills.\n\nExpected Output:\n{\n  title: 'Software Engineer',\n  company: 'Startup',\n  required_skills: ['Java', 'Git', 'Agile'],\n  preferred_skills: ['Docker', 'Kubernetes'],\n  yoe_range: '2-4 years',\n  salary_range: '$80k-$120k',\n  remote_policy: 'Remote',\n  match_percentage: 40.0,\n  missing_skills: ['Agile', 'Docker', 'Kubernetes']\n}"},
            {"role": "user", "content": f"Job Description:\n{job_description}\n\nJob Description 2:\n{job_description2}\n\nJob Description 3:\n{job_description3}\n\nResume:\n{resume}"}
        ],
        response_format=JobAnalysis,
        temperature=0
    )
    return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens

def compare_skills_chain_of_thought(job_description: str, job_description2: str, job_description3: str, resume: str) -> JobAnalysis:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Think step by step to extract job posting data. For title, use the exact job title only — no extra words. missing_skills should ONLY contain items from required_skills and preferred_skills that the candidate lacks. Do not invent new skills. Everything you do should be having a reason before you do it. For example, if you are extracting the required skills, you should first identify the section of the job description that lists the required skills, then extract the skills from that section. Do this for each field you need to extract."},
            {"role": "user", "content": f"Job Description:\n{job_description}\n\nJob Description 2:\n{job_description2}\n\nJob Description 3:\n{job_description3}\n\nResume:\n{resume}"}
        ],
        response_format=JobAnalysis,
        temperature=0
    )
    return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens

def compare_skills_structured_cot(job_description: str, job_description2: str, job_description3: str, resume: str) -> JobAnalysis:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "step 1: Identify the job title in the job description and extract it. step 2: Identify the company name in the job description and extract it. step 3: Identify the required skills in the job description and extract them into a list. step 4: Identify the preferred skills in the job description and extract them into a list. step 5: Identify the years of experience required in the job description and extract it. step 6: Identify the salary range in the job description and extract it. step 7: Identify the remote policy in the job description and extract it. step 8: Compare the required and preferred skills with the candidate's skills in the resume to calculate a match percentage. step 9: List any missing skills that are required or preferred but not present in the candidate's resume."},
            {"role": "user", "content": f"Job Description:\n{job_description}\n\nJob Description 2:\n{job_description2}\n\nJob Description 3:\n{job_description3}\n\nResume:\n{resume}"}
        ],
        response_format=JobAnalysis,
        temperature=0
    )
    return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens
def main():
    job_description3='''About the job
About Hebbia

The AI platform for investors and bankers that generates alpha and drives upside.

Founded in 2020 by George Sivulka and backed by Peter Thiel and Andreessen Horowitz, Hebbia powers investment decisions for BlackRock, KKR, Carlyle, Centerview, and 40% of the world's largest asset managers. Our flagship product, Matrix, delivers industry-leading accuracy, speed, and transparency in AI-driven analysis. It is trusted to help manage over $30 trillion in assets globally.

We deliver the intelligence that gives finance professionals a definitive edge. Our AI uncovers signals no human could see, surfaces hidden opportunities, and accelerates decisions with unmatched speed and conviction. We do not just streamline workflows. We transform how capital is deployed, how risk is managed, and how value is created across markets.

Hebbia is not a tool. Hebbia is the competitive advantage that drives performance, alpha, and market leadership.

The Role

We are seeking our first Data Engineer, someone who can refine our data infrastructure, drive best practices for building data pipelines, and collaborate closely with both engineering and business teams to ensure every data need is met. If you are a self-starter with a proven track record of architecting end-to-end data solutions, we'd love to hear from you.

Responsibilities

Architect, build, and maintain ETL pipelines and workflows that ensure high data quality and reliability
Design and manage a central data lake to consolidate data from various sources, enabling advanced analytics and reporting
Collaborate with cross-functional stakeholders (product, engineering, and business) to identify data gaps and develop effective solutions
Implement best practices in data security and governance to ensure compliance and trustworthiness
Evaluate and integrate new technologies, tools, and approaches to optimize data processes and architectures
Continuously monitor, troubleshoot, and improve data pipelines and infrastructure for performance, scalability, and cost-efficiency

Who You Are

Bachelor's or Master's degree in Computer Science, Data Science, Statistics, or a related field
5+ years software development experience at a venture-backed startup or top technology firm, with a focus on data engineering
Significant hands-on experience in data engineering (ETL development, data warehousing, data lake management, etc.)
Adept at identifying and owning data projects end to end, with the ability to work independently and exercise sound judgment
Proficient in Python and SQL; comfortable working with cloud-based data stack tools
Familiar with big data processing frameworks (e.g., Spark, Hadoop) and data integration technologies (e.g., Airflow, DBT, or similar)
Experience implementing data governance, security, and compliance measures
Strong collaboration and communication skills, with the ability to translate business requirements into technical solutions
Prior experience in a high-growth or startup environment is a plus
You are comfortable working in-person 5 days a week

Compensation

The salary range for this position is set between $190,000 and $250,000. This range may be inclusive of several career levels at Hebbia and will be narrowed during the interview process based on the candidate's experience and qualifications. Adjustments outside of this range may be considered for candidates whose qualifications significantly differ from those outlined in the job description.'''
    job_description2 = '''About the job
About Us

Notion helps you build beautiful tools for your life’s work. In today's world of endless apps and tabs, Notion provides one place for teams to get everything done, seamlessly connecting docs, notes, projects, calendar, and email—with AI built in to find answers and automate work. Millions of users, from individuals to large organizations like Toyota, Figma, and OpenAI, love Notion for its flexibility and choose it because it helps them save time and money.

In-person collaboration is essential to Notion's culture. We require all team members to work from our offices on Mondays, Tuesdays, and Thursdays, our designated Anchor Days. Certain teams or positions may require additional in-office workdays.

About The Role

We are looking for a Business Systems Engineer to join our Business Technology team to scale systems powering our GTM, Finance, and Operations teams. In this role, you'll design, build, and maintain integrations, automations, and workflows across our SaaS stack to ensure seamless data flows and reliable operations. Combining technical expertise with cross-functional collaboration, you'll partner with stakeholders in Sales, Finance, Marketing, and Support to deliver scalable solutions that drive business growth.

What You’ll Achieve

Systems Development & Customization
Design, build, and maintain solutions in Salesforce and other business-critical SaaS applications.
Configure workflows, automations, and data transformations to streamline processes and reduce manual work.
Integrations & Architecture
Build and maintain integrations using iPaaS tools (e.g., Workato, Tray.io, Celigo, Mulesoft).
Develop direct API integrations (Node.js, Python, AWS Lambda, REST/SOAP) for custom connectivity.
Contribute to long-term systems architecture and integration roadmap, ensuring scalability and resilience.
Data & Governance
Maintain high data quality and synchronization across CRM, ERP, billing, and support systems.
Create and maintain technical documentation, process flows, and best practices.
Collaboration & Delivery
Partner with GTM and Finance leaders to translate business requirements into scalable technical solutions.
Lead or contribute to cross-functional projects, from requirements gathering to implementation and support.
Troubleshoot and resolve integration and application issues, ensuring system reliability and uptime.
Skills You’ll Need To Bring

4–6 years of experience in Business Systems Engineering or Application Engineering in a SaaS environment.
Strong Salesforce development experience (Apex, SOQL, Lightning Web Components, APIs).
Proficiency with at least one programming or scripting language (Python, JavaScript, Node.js or TypeScript).
Experience with integration platforms (Workato, Tray.io, or Mulesoft).
Familiarity with ERP and GTM processes (e.g., lead-to-cash, quote-to-cash, customer lifecycle).
Strong understanding of data modeling, APIs, and system architecture.
Excellent communication skills; ability to work effectively with both technical and non-technical teams.
You don’t need to be an AI expert, but you’re curious and willing to adopt AI tools to work smarter and deliver better results

Nice To Haves

Experience building lightweight agents or automations with tools like Notion, scripting, or no-code platforms.
Working knowledge of sales automation and AI implementation

We hire talented and passionate people from a variety of backgrounds because we want our global employee base to represent the wide diversity of our customers. If you’re excited about a role but your past experience doesn’t align perfectly with every bullet point listed in the job description, we still encourage you to apply. If you’re a builder at heart, share our company values, and enthusiastic about making software toolmaking ubiquitous, we want to hear from you.

Notion is proud to be an equal opportunity employer. We do not discriminate in hiring or any employment decision based on race, color, religion, national origin, age, sex (including pregnancy, childbirth, or related medical conditions), marital status, ancestry, physical or mental disability, genetic information, veteran status, gender identity or expression, sexual orientation, or other applicable legally protected characteristic. Notion considers qualified applicants with criminal histories, consistent with applicable federal, state and local law. Notion is also committed to providing reasonable accommodations for qualified individuals with disabilities and disabled veterans in our job application procedures. If you need assistance or an accommodation due to a disability, please let your recruiter know.

Notion is committed to providing highly competitive cash compensation, equity, and benefits. The compensation offered for this role will be based on multiple factors such as location, the role’s scope and complexity, and the candidate’s experience and expertise, and may vary from the range provided below. For roles based in San Francisco and New York, the estimated base salary range for th'''


    job_description = '''
About the job
Who We Are

Verkada is transforming how organizations protect their people and places with an integrated, AI-powered platform. A leader in cloud physical security, Verkada helps organizations strengthen safety and efficiency through one connected software platform that includes solutions for video security, access control, air quality sensors, alarms, intercoms, and visitor management.

Over 30,000 organizations worldwide, including more than 100 companies in the Fortune 500, trust Verkada as their physical security layer for easier management, intelligent control, and scalable deployments. Founded in 2016, Verkada has expanded rapidly with 15 offices and 2,200+ full-time employees.

About The Role

Verkada’s Growth Ops team is looking for a talented individual to play an integral role in leveraging our data and technology to help our marketing team run better, faster, and smarter. You will have the opportunity to play a vital role in both creating and driving world-class insights that will drive the next phase of growth here at Verkada.

Successful Marketing Analytics leaders at Verkada are fluent in SQL and data modeling, understand the principles of effective data visualization, and know how to contextualize their work within the business in order to drive growth. This role reports to our Sr. Manager, Analytics Engineering. We are committed to a thriving in-office culture. This role requires that you be on-site at our office in San Mateo, CA 5 days a week.

What You'll Do

Own the full analytics workflow, from transforming raw data in dbt to building dashboards in Looker and presenting findings that influence strategy. 
Collaborate with stakeholders to design metrics that align with company objectives and inform decision-making. 
Build and maintain scalable, well-documented data models in dbt and Looker/LookML that power reporting and analysis across the org. 
Help shape data governance and foster data literacy, empowering stakeholders to self-serve with trustworthy and accurate data. 
Deliver high-impact, data-driven insights and recommendations; collaborate with leadership to influence key decisions and optimize the growth funnel

What You Bring

3–5 years of experience in data analytics or data science. 
Advanced SQL proficiency and experience working with modern databases (BigQuery, Snowflake, Redshift)
Experience building dashboards with data visualization tools (Looker, Tableau, Power BI). 
Demonstrated track record of success in a high-growth, self-directed business environment. 
Strong analytical mindset; enjoys solving complex and ambiguous problems. 
Excellent verbal and written communication skills, with ability to translate complex data into clear business narratives. 
Must be willing and able to commute 5 days in office. 

Nice to Have

Direct relevant experience in Marketing Analytics or a similar function (Growth Analytics, Data Science, Product Analytics, etc.)
Experience with data modeling, ETL/ELT, and data warehousing. 
Experience with dbt
Experience with Looker/LookML
Proficiency with Python, R, and statistical analysis. 
Familiarity with business systems: Salesforce, Marketo, Outreach, or similar. 

Employee Benefits

Verkada is committed to fostering a workplace environment that prioritizes the holistic health and wellbeing of our employees and their families by offering comprehensive wellness perks, benefits, and resources. Our benefits and perks programs include, but are not limited to:

Healthcare programs that can be tailored to meet the personal health and financial well-being needs - Premiums are 100% covered for the employee under at least one plan and 80% for family premiums under all plans Nationwide medical, vision and dental coverage
Health Saving Account (HSA) with annual employer contributions and Flexible Spending Account (FSA) with tax saving options
Expanded mental health support
Paid parental leave policy & fertility benefits
Time off to relax and recharge through our paid holidays, firmwide extended holidays, flexible PTO and personal sick time
Professional development stipend
Fertility Stipend
Wellness/fitness benefits
Healthy lunches provided daily
Commuter benefits

Additional Information

You must be independently authorized to work in the U.S. We are unable to sponsor or take over sponsorship of an employment visa for this role, at this time. 

Annual Pay Range

At Verkada, we want to attract and retain the best employees, and compensate them in a way that appropriately and fairly values their individual contribution to the company. With that in mind, we carefully consider a number of factors to determine the appropriate starting pay for an employee, including their primary work location and an assessment of a candidate's skills and experience, as well as market demands and internal parity. A Verkada employee may be eligible for additional forms of compensation, depending on their role, including sales incentives, discretionary bonuses, and/or equity in the company in the form of restricted stock units (RSUs)

Below is the annual on-target earnings (OTE) range for full-time employees for this position, comprised of base compensation and commissions (if applicable).

Estimated Annual Pay Range

$130,000—$180,000 USD

Verkada Is An Equal Opportunity Employer

As an equal opportunity employer, Verkada is committed to providing employment opportunities to all individuals. All applicants for positions at Verkada will be treated without regard to race, color, ethnicity, religion, sex, gender, gender identity and expression, sexual orientation, national origin, disability, age, marital status, veteran status, pregnancy, or any other basis prohibited by applicable law.'''
    resume = '''
SUMMARY
Manidweep Sharma 
Mankato, MN |manidweepsharmay@gmail.com| +1 (507)-613-1138 | LinkedIn | Portfolio 
• Data Engineer with 4+ years of experience in cloud-native data architectures, ETL pipeline development, 
and large-scale data processing across healthcare, finance, and education sectors.  
• AI Research: Lead evaluations of diffusion models vs. GANs for synthetic medical imaging, with ongoing research 
for an IEEE paper submission.  
• Financial Analytics: Developed AI simulators using Monte Carlo and Markov Chain methods to analyze member 
response under economic stress, reducing high-risk exposure for new product launches. 
• Cloud Engineering: Designed AWS/Azure pipelines (Glue, Snowflake, Databricks) for real-time data processing, 
cutting query costs and improving dataset efficiency. 
• Mentorship: Trained 50+ students in Python and SQL, emphasizing practical ML applications. Mentored 
undergraduates on soft skills/leadership. 
• Certifications: Microsoft Certified: Azure Data Engineer Associate, Microsoft Certified: Power BI Data Analyst 
Associate, AWS Certified: Machine Learning Engineer Associate 
SKILLS
Programming: Python, SQL, PySpark, Shell Scripting ETL Tools: Airflow, Azure Data Factory, dbt, AWS Glue, dlt 
Database: MySQL, PostgreSQL, Microsoft SQL Server, 
MongoDB 
Libraries: NumPy, Pandas, Scikit-learn, Seaborn, NLTK, PyMC 
Cloud Services: GCP (BigQuery, Dataflow), Azure (Synapse),  
Big Data: Apache Hadoop, Apache Spark, Kafka, Hive AWS (S3, EMR, QuickSight, Athena), Databricks 
Data Warehousing: Snowflake, ADLS, Redshift DevOps & CI/CD : Docker, GitHub 
Data Visualization: Power BI, Tableau Other: TensorFlow, ML, AI, Data Modeling, Data Transformation 
PROFESSIONAL EXPERIENCE
Data Researcher | Minnesota State University | Mankato, MN 
Aug 2024 – Present 
• Built automated pipelines for AI system evaluations comparing next-generation diffusion models against 
traditional GANs, demonstrating 40% higher-quality synthetic medical scans (X-rays/tissue samples) to enhance 
diagnostic tool training.  
• Validate synthetic data through statistical comparisons with real-world medical imaging datasets. 
• Mentored students in core programming (Python), database design, MIPS, and data analysis techniques through 
hands-on lab sessions. 
• Preparing an IEEE research paper comparing diffusion models and GANs in medical imaging, analyzing 10,000+ 
images and evaluating performance metrics. 
Data Scientist Intern | Affinity Plus Federal Credit Union | St Paul, MN    
Jun 2024 – Aug 2024 
• Collaborated with an Enterprise Digital Intelligence Data Science team to develop an AI-driven simulator, analyzing 
50,000+ member data points to predict behavior and internal variable fluctuations during economic distress.  
• Engineered scenario analyses with 10+ key internal variables to simulate customer decision-making, enabling 
product adjustments and improving forecast accuracy by 20%. 
• Delivered board-level presentations to C-suite executives, translating complex simulation outputs into strategic 
recommendations that improved decision-making speed by 30% and reduced high-risk product launches by 
20%. 
• Simulated datasets from scratch using Databricks and AWS (S3, Glue), aggregating 50+ internal variables (e.g., 
customer transactions, liquidity metrics) into time-series formats. 
Graduate Assistant (Data Engineer/Specialist) | Minnesota State University   
Aug 2023 – May 2024 
• Engineered scalable ETL pipelines (Python, SQL) to process 3000+ student retention datasets, integrating real
t
ime engagement metrics and historical academic records. 
• Built Power BI dashboards with automated dataflows and semantic modeling, improving enrollment trend 
detection by 30% through KPI tracking and Azure Synapse-integrated datasets.  
• Automated office workflows (Kearney International Center) using Python scripting and Automation tools, 
reducing manual data entry by 50%. 
• Mentored 20+ undergraduate students in the Global Leadership Academy, guiding personal project development 
to strengthen soft skills (communication, teamwork) and leadership competencies. 
Data Engineer | Infosys Limited | Hyderabad, India      
Mar 2021 – Dec 2022 
• Integrated payment gateways and APIs to consolidate 1M+ UK transaction records, improving data accuracy and 
efficiency. 
• Automated invoicing by harmonizing legacy and modern payment data into Snowflake via Azure Data Factory. 
• Deployed Azure Stream Analytics to process 500K+ real-time meter events daily, paired with Snowflake for trend 
analysis. 
• Ensured 99.9% accuracy in energy usage reporting through data validation for regulatory compliance. 
• Collaborated with backend team at E.ON UK to improve data retrieval efficiency by 35% using Azure Data Factory 
and Snowflake. 
• Led a team processing 7 years of customer data with Azure Data Factory pipelines, boosting productivity by 50% 
and dataset efficiency by 75%. 
PROJECTS
AI Simulator | Affinity Plus Federal Credit Union | St Paul, MN     
Jun 2024 – Aug 2024 
• Led development of an AI simulator using Monte Carlo simulations, Markov Chains, and probability distribution 
functions, achieving prediction accuracy improvements of 20% for new product market success using Python 
libraries like PyMC.  
• Created interactive dashboards with Power BI, streamlining data-driven decision-making and presenting insights 
to business teams. 
Big Data Recommendation System | Minnesota State University    
Aug 2024 – Dec 2024 
• Directed the development of a scalable recommendation system processing 5M+ records across distributed 
clusters, significantly enhancing personalized book suggestions and boosting customer engagement by 30%. 
• Architected and implemented an on-premises big data architecture using Spark and Hadoop, enabling high
performance analytics with 40% faster data processing and ensuring efficient data flow. 
AWS-Powered YouTube Data Analysis       
Jan 2025 – Feb 2025 
• Designed serverless ETL pipeline ingesting 2TB+ daily JSON data from YouTube API into AWS S3, transformed 
to Parquet using Glue ETL (PySpark) and AWS Wrangler, cutting Athena query costs by 60% through columnar 
storage optimization. 
• Automated metadata cataloging with Glue Crawlers, enabling schema-on-read for 50+ video metrics (views, 
engagement) and dynamic partitioning by upload date/category. 
• Created cost-monitoring dashboards in QuickSight using Athena SQL and Lambda-processed usage reports, 
identifying underperforming content with analysis of video titles/descriptions. 
EDUCATION
Minnesota State University | M.S in Data Science      
Geethanjali College of Engineering and Technology       
ORGANIZATIONS
Jan 2023 – Dec 2024 
Aug 2016 – Oct 2020 
President, Student Association of India       
Aug 2023 – May 2024 
Member, DREAM (Data Resources for Eager & Analytical Minds) '''

    analysis_zero_shot = compare_skills_zero_shot(job_description,job_description2,job_description3, resume)
    analysis_few_shot = compare_skills_few_shot(job_description,job_description2,job_description3, resume)
    analysis_cot = compare_skills_chain_of_thought(job_description,job_description2,job_description3, resume)
    analysis_structured_cot = compare_skills_structured_cot(job_description,job_description2,job_description3, resume)
    # create list of the three job descriptions
    jds = [job_description, job_description2, job_description3]

    # wrappers to call existing compare functions for a single JD
    def zero_shot_single(jd, resume):
        return compare_skills_zero_shot(jd, "", "", resume)

    def few_shot_single(jd, resume):
        return compare_skills_few_shot(jd, "", "", resume)

    def structured_cot_single(jd, resume):
        return compare_skills_structured_cot(jd, "", "", resume)
    
    def compare_skills_chain_of_thought_single(jd, resume):
        return compare_skills_chain_of_thought(jd, "", "", resume)

    methods = [
        ("zero_shot", zero_shot_single),
        ("few_shot", few_shot_single),
        ("chain_of_thought", compare_skills_chain_of_thought_single),
        ("structured_cot", structured_cot_single),
    ]

    # iterate over each JD and method, measuring latency and printing metrics in a table
    for i, jd in enumerate(jds, start=1):
        print(f"\n=== Job {i} ===")
        print(f"{'Strategy':<20} {'Match %':<10} {'Prompt Tokens':<15} {'Completion Tokens':<18} {'Latency'}")
        print("-" * 80)
        for name, func in methods:
            start = time.perf_counter()
            try:
                parsed, prompt_tokens, completion_tokens, total_tokens = func(jd, resume)
            except Exception as e:
                print(f"{name:<20} FAILED     {e}")
                continue
            end = time.perf_counter()
            latency = end - start

            match_pct = getattr(parsed, "match_percentage", None)
            if match_pct is None:
                match_display = 'N/A'
            else:
                match_display = f"{match_pct}"

            missing = None
            try:
                missing = getattr(parsed, "missing_skills", None)
                if missing is None and hasattr(parsed, "dict"):
                    missing = parsed.dict().get("missing_skills")
            except Exception:
                missing = None
            missing_count = len(missing) if isinstance(missing, (list, tuple)) else (0 if missing is None else 1)

            print(f"{name:<20} {match_display:<10} {prompt_tokens:<15} {completion_tokens:<18} {latency:.2f}s")

if __name__ == "__main__":
    main()