import re
import openai
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai_api_key)

def extract_job_details(job_description) -> dict:
    prompt = f"""
    Extract the following details from the job description:
    1. Job Title
    2. Company Name
    3. Location
    4. Required Skills
    5. Experience Level

    Job Description: {job_description}

    Please provide the details in JSON format.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts job details from job descriptions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content.strip()
    try:
        # Try direct JSON parsing first
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response using regex
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print("Failed to parse JSON response. Raw response:")
        print(content)
        return {}
    
def compare_skills(job_description, candidate_skills):
    details = extract_job_details(job_description)
    required_skills = set(details.get("Required Skills", []))
    candidate_skills_set = set(candidate_skills)

    missing_skills = required_skills - candidate_skills_set
    matched_skills = required_skills & candidate_skills_set

    return {
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills)
    }

def extract_skills_from_resume(resume_text):
    prompt = f"""
    Extract the skills from the following resume text and return them in JSON format with a "skills" key containing an array of skills.

    Resume Text: {resume_text}

    Return format example:
    {{"skills": ["Python", "SQL", "Machine Learning", ...]}}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts skills from resumes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content.strip()
    try:
        print("Raw response for skills extraction:")
        print(content)
        result = json.loads(content)
        # Return the list of skills, not the whole dict
        return result.get("skills", [])
    except json.JSONDecodeError:
        # Try to extract JSON from the response using regex
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return result.get("skills", [])
            except json.JSONDecodeError:
                pass
        print("Failed to parse JSON response. Raw response:")
        print(content)
        return []
def analyze_my_resume(resume_text, job_description):
    # Extract skills from the resume (returns a list)
    resume_skills = extract_skills_from_resume(resume_text)

    # Compare resume skills with job description requirements
    comparison = compare_skills(job_description, resume_skills)

    return {
        "resume_skills": resume_skills,
        "matched_skills": comparison["matched_skills"],
        "missing_skills": comparison["missing_skills"]
    } 

# Example usage
if __name__ == "__main__":
    job_description = """
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

As an equal opportunity employer, Verkada is committed to providing employment opportunities to all individuals. All applicants for positions at Verkada will be treated without regard to race, color, ethnicity, religion, sex, gender, gender identity and expression, sexual orientation, national origin, disability, age, marital status, veteran status, pregnancy, or any other basis prohibited by applicable law.
    """

    resume_text= """
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
Member, DREAM (Data Resources for Eager & Analytical Minds) 


    """
    analysis_result = analyze_my_resume(resume_text, job_description)
    print("\n" + "="*60)
    print("RESUME ANALYSIS RESULT")
    print("="*60)
    print(f"\nResume Skills Found ({len(analysis_result['resume_skills'])}):")
    for skill in analysis_result['resume_skills']:
        print(f"  • {skill}")

    print(f"\nMatched Skills ({len(analysis_result['matched_skills'])}):")
    for skill in analysis_result['matched_skills']:
        print(f"  ✓ {skill}")

    print(f"\nMissing Skills ({len(analysis_result['missing_skills'])}):")
    for skill in analysis_result['missing_skills']:
        print(f"  ✗ {skill}")

    print("\n" + "="*60)
    print(f"Match Rate: {len(analysis_result['matched_skills'])}/{len(analysis_result['matched_skills']) + len(analysis_result['missing_skills'])} ({len(analysis_result['matched_skills']) / (len(analysis_result['matched_skills']) + len(analysis_result['missing_skills'])) * 100:.1f}%)")
    print("="*60)