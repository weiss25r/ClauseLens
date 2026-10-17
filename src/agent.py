from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

class DocumentInspectorAgent:
    def __init__(self, model_name: str, policy:str):
        system_prompt = """You are a legal document inspector.
        You will be given a paragraph, its label, an internal company policy and an output format. Your task is to determine if the paragraph is compliant with the policy.
        If the paragraph is not compliant, you will provide a reason for the non-compliance and rewrite the paragraph according to the policy.
        Otherwise, you will say that the paragraph is compliant. Your output should be a JSON with keys: paragraph, reason, corrections."""

        human_prompt = """
        Input Paragraph: {paragraph}
        Paragraph label: {label}
        Policy: {policy}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])


        print(model_name)
        self.llm = ChatOllama(model=model_name, format='json')
        self.policy = policy
        self.parser = JsonOutputParser()

        self.chain = prompt | self.llm | self.parser


    def run(self, chunks, labels):
        llm_output = []

        for paragraph, label in zip(chunks, labels):
            if label == 'Other':
                continue

            output = self.chain.invoke(
                {
                    "paragraph": paragraph,
                    "label": label,
                    "policy": self.policy
                }
            )

            llm_output.append(output)
        return llm_output