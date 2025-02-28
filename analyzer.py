import os
import sys
import re
import json
from pathlib import Path
import fitz 
from typing import Dict, List, Tuple, Set, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

class RAGResumeAnalyzer:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY", "your_api_key_here")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,  
            max_tokens=1500
        )
        
        # Initialize cache directory
        cache_dir = Path.home() / ".resume_analyzer_cache"
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir
        
        # Company requirements
        self.required_experience_years = 2
        self.required_skills_categories = {
            "programming": ["sql", "python", "java", "javascript"],
            "math_stats": ["statistics", "calculus", "probability", "linear algebra", "regression"],
            "machine_learning": ["tensorflow", "pytorch", "scikit-learn", "ml", "machine learning", "supervised learning", 
                                "unsupervised learning", "svm", "decision tree", "random forest"],
            "data_visualization": ["tableau", "power bi", "matplotlib", "visualization", "dash", "plotly", "d3", 
                                  "seaborn", "ggplot"],
            "data_wrangling": ["data transformation", "data cleaning", "wrangling", "ETL", "data preparation", 
                              "data integration", "data modeling"],
            "data_mining": ["information extraction", "data mining", "insights",  
                           "data profiling", "pattern recognition"],
            "nlp": ["natural language processing", "text analysis", "nlp", "translation", "named entity recognition", 
                   "sentiment analysis", "text classification", "tokenization"],
            "deep_learning": ["neural networks", "deep learning", "cnn", "rnn", "lstm", "gan", "transformer", 
                             "image processing", "computer vision"],
            "time_series": ["time series", ],
            "llm": ["large language models", "LLM", "generative ai", "transformer", "llama", "openai"],
            "agentic_ai": ["agent", "agentic ai", "rag", "retrieval augmented generation", 
                          "multi-agent", "cognitive", "reasoning"],
            "langchain": ["langchain", "agents", "chains", "tools", "llamaindex", "vector database", "faiss", 
                         "pinecone", "chromadb", "qdrant"]
        }
        
        # Initialize RAG system upon creation
        self.resume_text = None
        self.resume_hash = None
        self.vector_store = None
        self.conversation_chain = None
    
    def setup_rag_system(self, resume_text: str):
        """Set up RAG system for both analysis and Q&A"""
        self.resume_text = resume_text
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300
        )
        chunks = text_splitter.split_text(resume_text)
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=memory
        )
        
        return self.conversation_chain

    def extract_candidate_details(self) -> Dict:
        """Extract candidate details using structured prompting"""
        query = """
        Please extract ONLY the following information from the resume:
        1. Full name of the candidate
        2. Contact number/phone
        3. Email address
        4. Physical address/location
        5. Any professional links (LinkedIn, GitHub, portfolio, etc.)
        
        Format your response EXACTLY as:
        Name: [full name]
        Phone: [phone number or "Not found"]
        Email: [email or "Not found"]
        Address: [address or "Not found"]
        Links: [comma-separated list of links or "None"]
        
        If any information is not present in the resume, use "Not found" or "None" for Links.
        """
        
        try:
            response = self.conversation_chain({"question": query})
            response_text = response['answer']
            
            details = {}
            
            # Extract name
            name_match = re.search(r"Name:\s*(.*?)(?:\n|$)", response_text)
            details["name"] = name_match.group(1).strip() if name_match else "Not found"
            
            # Extract phone
            phone_match = re.search(r"Phone:\s*(.*?)(?:\n|$)", response_text)
            details["phone"] = phone_match.group(1).strip() if phone_match else "Not found"
            
            # Extract email
            email_match = re.search(r"Email:\s*(.*?)(?:\n|$)", response_text)
            details["email"] = email_match.group(1).strip() if email_match else "Not found"
            
            # Extract address
            address_match = re.search(r"Address:\s*(.*?)(?:\n|$)", response_text)
            details["address"] = address_match.group(1).strip() if address_match else "Not found"
            
            # Extract links
            links_match = re.search(r"Links:\s*(.*?)(?:\n|$)", response_text)
            links_str = links_match.group(1).strip() if links_match else ""
            if links_str.lower() in ["not found", "none"]:
                details["links"] = []
            else:
                details["links"] = [link.strip() for link in links_str.split(",") if link.strip()]
            
            return details
        except Exception as e:
            print(f"Error extracting candidate details: {str(e)}")
            return {
                "name": "Not found",
                "phone": "Not found",
                "email": "Not found",
                "address": "Not found",
                "links": []
            }

    def extract_experience_and_skills(self) -> Tuple[float, List[str]]:
        """Extract years of experience and skills using improved structured prompting"""
        # First, attempt to extract skills with a dedicated query
        skills_query = """
        Please analyze the resume and extract ALL technical skills, technologies, programming languages, 
        frameworks, methodologies, and domain knowledge mentioned.
        
        Format ONLY as a comma-separated list:
        [skill1], [skill2], [skill3], ...
        
        DO NOT include any explanations, headers, or other text.
        """
        
        skills_response = self.conversation_chain({"question": skills_query})
        skills_text = skills_response['answer'].strip()
        
        # Clean up skills list
        if skills_text.startswith("[") and skills_text.endswith("]"):
            skills_text = skills_text[1:-1]
        
        skills = [skill.strip().lower() for skill in skills_text.split(",") if skill.strip()]
        
        # Now, get a more reliable experience extraction with a better prompt
        exp_query = """
        Please analyze the resume carefully and determine the total years of professional experience.
        
        Instructions:
        1. Add up all relevant work experience periods
        2. Include all professional roles including internships (but count internships at 0.5 weight)
        3. DO NOT include education years unless they were combined with professional work
        4. If someone has X years Y months, express as X.Y/12 (e.g., 3 years 6 months = 3.5)
        5. If you find ANY work experience, the value CANNOT be zero
        
        Format your response EXACTLY as a single number followed by explanation:
        [years] - [brief justification]
        
        Example: "4.5 - 2 years at Company A, 1.5 years at Company B, 1 year consulting"
        """
        
        exp_response = self.conversation_chain({"question": exp_query})
        exp_text = exp_response['answer'].strip()
        
        # Extract just the number from the response
        years_match = re.search(r"^(\d+\.?\d*)", exp_text)
        if years_match:
            years = float(years_match.group(1))
        else:
            # Fallback: try to find any number followed by "years" or "year"
            years_match = re.search(r"(\d+\.?\d*)\s*years?", exp_text)
            years = float(years_match.group(1)) if years_match else 0
        
        # Validate - if we have significant skills but zero experience, likely an issue
        if years == 0 and len(skills) > 5:
            # Try one more time with an explicit focus
            retry_query = """
            I need to calculate the EXACT total years of work experience from this resume.
            Evidence suggests this candidate has skills but the experience wasn't properly calculated.
            
            Please re-examine ALL dates, roles, and positions mentioned.
            Even short-term positions, part-time roles, and projects count.
            
            Format ONLY as a number: [years]
            """
            
            retry_response = self.conversation_chain({"question": retry_query})
            retry_text = retry_response['answer'].strip()
            
            # Try to extract just the number
            retry_match = re.search(r"(\d+\.?\d*)", retry_text)
            if retry_match:
                years = float(retry_match.group(1))
        
        return years, skills

    def match_skills(self, keywords: List[str]) -> Dict[str, Dict]:
        """Match extracted keywords against required skill categories"""
        matches = {}
        all_matched_skills = set()
        
        for category, skills in self.required_skills_categories.items():
            matched_skills = set()
            
            for skill in skills:
                # Look for exact matches or skill as substring of keyword or vice versa
                for keyword in keywords:
                    # Check for exact matches
                    if skill.lower() == keyword.lower():
                        matched_skills.add(skill)
                        all_matched_skills.add(keyword)
                    # Check for substring matches
                    elif skill.lower() in keyword.lower() or keyword.lower() in skill.lower():
                        matched_skills.add(skill)
                        all_matched_skills.add(keyword)
            
            # Store both boolean match status and the matched skills
            matches[category] = {
                "matched": len(matched_skills) > 0,
                "skills": list(matched_skills)
            }
        
        # Also identify unmatched keywords that might be relevant
        unmatched_keywords = set(keywords) - all_matched_skills
        
        return matches, list(unmatched_keywords)

    def generate_hr_assessment(self, name: str, years_exp: float, matching_categories: int, 
                            matched_skill_list: List[str], keywords: List[str], 
                            unmatched_keywords: List[str]) -> str:
        """Generate HR assessment using LLM with improved prompt engineering"""
        prompt = f"""As an expert HR professional, provide a detailed assessment of this candidate:

        Name: {name}
        Years of Experience: {years_exp:.1f} years
        Required Experience: {self.required_experience_years}+ years
        
        Matching Skill Categories: {matching_categories}/12
        
        Matched Skills by Category:
        {chr(10).join(matched_skill_list)}
        
        All Extracted Keywords: {", ".join(keywords)}
        
        Unmatched Keywords: {", ".join(unmatched_keywords)}
        
        Instructions:
        1. Provide a comprehensive professional assessment including:
           - Evaluation of their core skills and experience
           - Specific mention of their experience with any LLM, Python, Machine Learning skills
           - Whether they meet the minimum requirements ({self.required_experience_years}+ years)
           - Areas of strength
           - Any potential gaps in skills
           - Clear hiring recommendation
        
        2. Be factual and based only on the provided data
           - If the candidate has {years_exp:.1f} years experience, focus on that number
           - If matching categories is {matching_categories}/12, use that exact number
        
        3. Format your response as a proper HR assessment (10-15 sentences)
        
        4. Be specific about the skills found in the resume
        """
        
        return self.llm.invoke(prompt).content

    def analyze_resume(self) -> Dict:
        """Main function to analyze a resume using RAG for extraction and LLM for assessment"""
        if not self.conversation_chain:
            raise Exception("RAG system not initialized. Call setup_rag_system first.")
        
        try:
            # Extract candidate details
            candidate_details = self.extract_candidate_details()
            
            # Extract information using RAG
            years_exp, keywords = self.extract_experience_and_skills()
            
            # Match skills using traditional approach
            skill_matches, unmatched_keywords = self.match_skills(keywords)
            
            # Calculate match score
            matching_categories = sum(1 for cat in skill_matches.values() if cat["matched"])
            meets_experience = years_exp >= self.required_experience_years
            
            # Format matched skills for HR assessment
            matched_skill_list = []
            for category, match_info in skill_matches.items():
                if match_info["matched"]:
                    category_name = category.replace("_", " ").title()
                    skills_str = ", ".join(match_info["skills"])
                    matched_skill_list.append(f"{category_name}: {skills_str}")
            
            # Generate HR assessment with LLM
            hr_assessment = self.generate_hr_assessment(
                candidate_details["name"], years_exp, matching_categories, matched_skill_list, 
                keywords, unmatched_keywords
            )
            
            result = {
                "candidate_details": candidate_details,
                "years_experience": years_exp,
                "matching_categories": matching_categories,
                "meets_experience": meets_experience,
                "skill_matches": skill_matches,
                "extracted_keywords": keywords,
                "unmatched_keywords": unmatched_keywords,
                "hr_assessment": hr_assessment,
                "recommended": matching_categories >= 6 and meets_experience,
                "matched_skill_list": matched_skill_list
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file with improved error handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text() + "\n"
        
        doc.close()
        
        if len(text.strip()) < 100:
            # Attempt to extract text with OCR fallback if available
            raise Exception("Extracted text is too short. PDF may be image-based or corrupted.")
            
        return text
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

def main():
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = RAGResumeAnalyzer()
    
    print("\n=== Resume Analysis System ===")
    print("Enter the path to your resume PDF:")
    resume_path = input("> ").strip()
    
    # Validate file exists
    if not os.path.exists(resume_path):
        print(f"Error: File not found at {resume_path}")
        sys.exit(1)
    
    # Extract text from PDF
    try:
        print("Extracting text from PDF...")
        resume_text = extract_text_from_pdf(resume_path)
        print(f"Text extraction complete. Extracted {len(resume_text)} characters.")
        
        # Initialize RAG system
        print("Setting up RAG system...")
        analyzer.setup_rag_system(resume_text)
        print("RAG system initialized successfully.")
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        sys.exit(1)
    
    while True:
        print("\n=== Resume Options ===")
        print("1. Ask questions about the resume")
        print("2. Get HR assessment")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n=== Resume Q&A Mode ===")
            print("You can now ask questions about the resume.")
            print("Type 'back' to return to the previous menu.\n")
            
            while True:
                question = input("\nAsk a question about the resume: ")
                
                if question.lower() == 'back':
                    break
                
                if not question.strip():
                    continue
                    
                print("Analyzing...")
                try:
                    response = analyzer.conversation_chain({"question": question})
                    print("\nAnswer:", response['answer'])
                except Exception as e:
                    print(f"Error: {str(e)}")
                
        elif choice == "2":
            # Run HR assessment
            print("\nGenerating HR assessment...")
            result = analyzer.analyze_resume()
            
            if "error" in result:
                print(f"\nError: {result['error']}")
                continue
                
            print("\n=== HR Assessment Results ===")
            candidate_details = result['candidate_details']
            print(f"Candidate Name: {candidate_details['name']}")
            print(f"Phone: {candidate_details['phone']}")
            print(f"Email: {candidate_details['email']}")
            print(f"Address: {candidate_details['address']}")
            
            if candidate_details['links']:
                print("Links:")
                for link in candidate_details['links']:
                    print(f"- {link}")
            
            print(f"\nYears of Experience: {result['years_experience']:.1f}")
            print(f"Matching Skill Categories: {result['matching_categories']}/12")
            print(f"\nExtracted Skills: {', '.join(result['extracted_keywords'])}")
            
            print("\nHR Assessment:")
            print(result['hr_assessment'])
            
            print(f"\nRecommendation: {'Proceed with interview' if result['recommended'] else 'Does not meet minimum requirements'}")
                
        elif choice == "3":
            print("Exiting resume analyzer. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()