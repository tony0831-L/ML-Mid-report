class gpt_info:
    openai_model: str
    temperature: int
    max_attempts: int
    attempts: int

    def __init__(self, openai_model: str, temperature: int, max_attempts: int, attempts: int):
        self.openai_model = openai_model
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.attempts = attempts
