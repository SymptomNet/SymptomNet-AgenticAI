from flask_restful import Resource
from flask import request, jsonify
from google import genai
from google.genai import types
from util.common import gemini_api

client = genai.Client(api_key=gemini_api)

class geminiPOSTResource(Resource):
    def post(self):
        message = request.json['Message']
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="You are a doctor that has extensive medical releated knowledge to figure out how to diagnose an illness given a set of symptoms provided by the patient or user. \
                                For every input you receive, you should produce a summary of the steps to diagnose an illness that spanse no more than 100 words. The steps must be in the format of a list of strings \
                                    for example: [\"Step 1: text\", \"Step 2: text\"]"),
            contents=message
        )
        
        return jsonify({"result": response.text})