few_shot_prompt = """
Question: A conference room can seat 12 people. 7 have arrived, then 3 groups of 4 each come. How many more chairs are needed?
Reasoning: 7 seated, 5 seats left. 3x4=12 new people need seats. 12 seats needed, 5 available, 12-5=7 more chairs
Answer: 7 chairs

Question: {new_question}
Reasoning:
"""

if __name__ == "__main__":
    print(few_shot_prompt)