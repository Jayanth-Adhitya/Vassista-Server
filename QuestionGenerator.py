from typing import List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

class QuestionGenerator:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(
            """
You are an expert interview question generator. Given the interview type, candidate resume, and a cover letter/background, generate a list of high-quality, relevant interview questions. Return only the list of questions, numbered.

Interview Type: {interview_type}
Resume: {resume}
Cover Letter/Background: {cover_letter}
{background_story_section}Generate {num_questions} interview questions tailored to this candidate and role. Return only the questions, numbered.
"""
        )

    async def generate_questions(
        self,
        interview_type: str,
        resume: str,
        cover_letter: str,
        background_story: Optional[str] = None,
        num_questions: int = 8
    ) -> List[str]:
        background_story_section = f"Background Story: {background_story}\n" if background_story else ""
        prompt = self.prompt_template.format(
            interview_type=interview_type,
            resume=resume,
            cover_letter=cover_letter,
            background_story_section=background_story_section,
            num_questions=num_questions
        )
        response = await self.llm.ainvoke(prompt)
        # Parse numbered questions from response
        questions = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                # Remove number/bullet
                q = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                questions.append(q)
            elif line:
                questions.append(line)
        return [q for q in questions if q]

# Example Gemini 2.0 Flash instantiation (swap as needed)
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_gemini_question_generator():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    return QuestionGenerator(llm)
