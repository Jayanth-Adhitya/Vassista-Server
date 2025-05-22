import asyncio
import os
from QuestionGenerator import get_gemini_question_generator

async def main():
    generator = get_gemini_question_generator()
    interview_type = "Software Engineer"
    resume = "Experienced Python developer with 5 years in backend systems, cloud, and APIs. Led a team at Acme Corp."
    cover_letter = "I am passionate about scalable software and have a strong background in distributed systems."
    background_story = "Recently completed a major migration project to cloud-native infrastructure."
    questions = await generator.generate_questions(
        interview_type=interview_type,
        resume=resume,
        cover_letter=cover_letter,
        background_story=background_story,
        num_questions=5
    )
    print("Generated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

if __name__ == "__main__":
    asyncio.run(main())
