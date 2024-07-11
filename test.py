import google. generativeai as genai
import os
os.environ['GOOGLE_API_KEY']="AIzaSyAFvL-vX0m67T1aglNrwDdsov5JWQcZqTc"

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("What are you?")

print(response.text)