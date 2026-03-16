from langchain_ollama.chat_models import ChatOllama

class DocumentInspectorAgent:
    def __init__(self, model_name: str, policy:str):
        system_prompt = """You are a legal document inspector.
        You will be given a paragraph, its label, an internal company policy and an output format. Your task is to determine if the paragraph is compliant with the policy.
        If the paragraph is not compliant, you will provide a reason for the non-compliance and rewrite the paragraph according to the policy.
        Otherwise, you will say that the paragraph is compliant. Your output should be a JSON with keys: paragraph, reason, corrections."""

        prompt_template = """
        Input Paragraph: {paragraph}
        Paragraph label: {label}
        Policy: {policy}
        """

        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.llm = ChatOllama(model=model_name, format='json')
        self.policy = policy

    def run(self, chunks, labels):
        llm_output = []

        for paragraph, label in zip(chunks, labels):
            if label == 'Other':
                continue
            messages = [
                (
                    "system",
                    self.system_prompt
                ),
                (
                    "human",
                    self.prompt_template.format(paragraph=paragraph, label=label, policy=self.policy)
                )
            ]
            output = self.llm.invoke(messages).content
            llm_output.append(output)
        return llm_output