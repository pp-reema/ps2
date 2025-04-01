import os
import sys
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class RelationshipInsightsGenerator:
    def __init__(self):
        """Initialize the relationship insights generator with LLM components."""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
        )
        
        self.relationship_template = """
        Generate brief relationship insights not more than 5 sentences for the MBTI personality type {mbti_type}:

        Context: Provide a nuanced overview of how {traits} individuals approach relationships, 
        emotional connections, and personal interactions. Consider their core personality 
        characteristics and how these manifest in romantic, professional, and personal relationships.

        Output Format:
        1. A brief paragraph exploring relationship dynamics
        2. A list of key relationship strengths
        3. A list of potential relationship challenges

        Focus on:
        - Brief overview of relationship approach
        - Top 3 relationship strengths
        - Top 3 relationship challenges

        Tone: Insightful, empathetic, and constructive
        """
        
        self.prompt = PromptTemplate(
            input_variables=["mbti_type", "traits"],
            template=self.relationship_template
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        # MBTI type traits and relationship characteristics
        self.mbti_relationship_traits = {
            "ISTJ": "logical, structured, and reliability-focused",
            "ISFJ": "nurturing, loyal, and detail-oriented",
            "INFJ": "deeply empathetic, intuitive, and idealistic",
            "INTJ": "strategic, independent, and intellectually driven",
            "ISTP": "pragmatic, adaptable, and action-oriented",
            "ISFP": "artistic, sensitive, and spontaneous",
            "INFP": "idealistic, empathetic, and authenticity-seeking",
            "INTP": "analytical, curious, and conceptually complex",
            "ESTP": "energetic, spontaneous, and action-focused",
            "ESFP": "enthusiastic, warm, and experience-driven",
            "ENFP": "creative, passionate, and connection-oriented",
            "ENTP": "innovative, challenging, and intellectually stimulating",
            "ESTJ": "organized, decisive, and goal-oriented",
            "ESFJ": "supportive, harmonious, and people-pleasing",
            "ENFJ": "charismatic, empathetic, and nurturing",
            "ENTJ": "ambitious, direct, and leadership-driven"
        }
    
    def generate_relationship_insights(self, mbti_type, conversation=None):
        """
        Generate personalized relationship insights based on MBTI type.
        
        Args:
            mbti_type (str): The MBTI personality type
            conversation (optional): Conversation context for additional personalization
        
        Returns:
            str: Brief relationship insights not more than 5 sentences
        """
        if mbti_type not in self.mbti_relationship_traits:
            raise ValueError(f"Invalid MBTI type: {mbti_type}")
        
        traits = self.mbti_relationship_traits[mbti_type]
        
        # Generate insights using LLM
        result = self.chain.run(mbti_type=mbti_type, traits=traits)
        
        # Parse and format the result for better readability
        return self._format_relationship_insights(result, mbti_type)
    
    def _format_relationship_insights(self, insights, mbti_type):
        """
        Format the raw insights into a structured, readable format.
        
        Args:
            insights (str): Raw insights text
            mbti_type (str): MBTI personality type
        
        Returns:
            str: Formatted relationship insights
        """
        # Additional formatting and structure can be added here
        formatted_insights = f"""
**Relationship Insights for {mbti_type}**

{insights}

"""
        return formatted_insights
