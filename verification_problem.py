from dotenv import load_dotenv
import os

load_dotenv()
print("Clé chargée :", os.getenv("GROQ_API_KEY"))
