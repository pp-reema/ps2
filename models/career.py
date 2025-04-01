import os
import sys
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class CareerInsightsGenerator:
    def __init__(self):
        """Initialize the career insights generator with LLM components."""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
        )
        
        self.career_template = """
        Generate brief career insights for the MBTI personality type {mbti_type}:

        Provide a shallow dive into career dynamics, work preferences, professional strengths, 
        and potential growth areas for {traits} individuals.

        Explore the following aspects:
        - Professional environment preferences
        - Ideal work settings and cultures
        - Motivational triggers
        - Potential career paths
        - Workplace strengths and challenges
        - Learning and development approaches
        - Decision-making in professional contexts
        - Potential areas of professional growth

        Consider both technical and soft skill dimensions.
        Provide actionable, inspiring career guidance.
        
        Output should be nuanced, motivational, and tailored to the specific MBTI type's 
        unique cognitive and emotional landscape not more than 5 sentences.
        """
        
        self.prompt = PromptTemplate(
            input_variables=["mbti_type", "traits"],
            template=self.career_template
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        # MBTI type traits and career characteristics
        self.mbti_career_traits = {
            "ISTJ": "methodical, detail-oriented, and systems-focused",
            "ISFJ": "supportive, meticulous, and service-driven",
            "INFJ": "insightful, strategic, and purpose-driven",
            "INTJ": "strategic, analytical, and innovation-oriented",
            "ISTP": "practical, adaptable, and problem-solving",
            "ISFP": "creative, sensitive, and aesthetically inclined",
            "INFP": "idealistic, empathetic, and values-centered",
            "INTP": "innovative, analytical, and conceptually complex",
            "ESTP": "dynamic, pragmatic, and action-oriented",
            "ESFP": "energetic, spontaneous, and people-focused",
            "ENFP": "creative, enthusiastic, and possibility-driven",
            "ENTP": "innovative, entrepreneurial, and challenge-seeking",
            "ESTJ": "organized, decisive, and efficiency-driven",
            "ESFJ": "collaborative, supportive, and harmony-seeking",
            "ENFJ": "inspirational, empathetic, and leadership-oriented",
            "ENTJ": "strategic, ambitious, and results-focused"
        }
    
    def generate_career_insights(self, mbti_type, conversation=None):
        """
        Generate personalized career insights based on MBTI type.
        
        Args:
            mbti_type (str): The MBTI personality type
            conversation (optional): Conversation context for additional personalization
        
        Returns:
            str: Brief career insights not more than 5 sentences
        """
        if mbti_type not in self.mbti_career_traits:
            raise ValueError(f"Invalid MBTI type: {mbti_type}")
        
        traits = self.mbti_career_traits[mbti_type]
        
        # Generate insights using LLM
        result = self.chain.run(mbti_type=mbti_type, traits=traits)
        
        # Parse and format the result for better readability
        return self._format_career_insights(result, mbti_type)
    
    def _format_career_insights(self, insights, mbti_type):
        """
        Format the raw insights into a structured, readable format.
        
        Args:
            insights (str): Raw insights text
            mbti_type (str): MBTI personality type
        
        Returns:
            str: Formatted career insights
        """
        # Additional structuring and insight generation
        formatted_insights = f"""
**Your Perfect Career**

{insights}

**Professional Profile for {mbti_type}**
**Strengths**
- Analytical mindset
- Strategic thinking
- Innovative problem-solving
- Deep focus and concentration

**Growth Areas**
- Flexibility in approach
- Interpersonal collaboration
- Emotional intelligence
- Adaptive learning
"""
        return formatted_insights

    def _generate_career_paths(self, mbti_type):
        """
        Generate potential career paths based on MBTI type.
        
        Args:
            mbti_type (str): The MBTI personality type
        
        Returns:
            list: Recommended career paths
        """
        career_recommendations = {
            "ISTJ": [
                "Accountant", 
                "Data Analyst", 
                "Project Manager", 
                "Systems Administrator"
            ],
            "INTJ": [
                "Data Scientist", 
                "Strategic Consultant", 
                "Research Scientist", 
                "Technology Architect"
            ],
            
            # ... (would include all 16 types)
        }
        
        return career_recommendations.get(mbti_type, [])
