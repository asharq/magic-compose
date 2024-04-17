import streamlit as st
import boto3
from botocore.config import Config
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the AWS Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    config=Config(read_timeout=60, connect_timeout=60, retries={'max_attempts': 10, 'mode': 'adaptive'})
)

def craft_prompt(message, feature, detail):
    """ Generate the appropriate prompt for each feature """
    if feature == 'Change writing tone':
        return f"Human: Please transform the following message according to the specified style: '{detail}'. Return only the transformed text: '{message}'\n\nAssistant:"
    elif feature == 'Spelling and grammar':
        return f"Human: Please correct any spelling and grammar mistakes in the following message and return only the corrected text: '{message}'\n\nAssistant:"
    elif feature == 'Shorten/elaborate':
        action = "expand with more details" if detail == "elaborate" else "shorten"
        return f"Human: Please {action} the following message and return only the text: '{message}'\n\nAssistant:"
    elif feature == 'Translation':
        return f"Human: Please translate the following message into {detail} and return only the translated text: '{message}'\n\nAssistant:"
    elif feature == 'Expand My Writing':
        return f"Human: Please generate three possible continuations for the following message: '{message}'\n\nAssistant:"
    elif feature == 'Analyze My Writing':
        return f"Human: Please analyze the following message and provide detailed feedback and suggestions for improvement, and also return a revised version of the text: '{message}'\n\nAssistant:"

def send_request_to_bedrock(message, feature, detail):
    """ Send the crafted prompt to AWS Bedrock and return the model's response """
    prompt = craft_prompt(message, feature, detail)
    logging.info("Sending prompt to model: %s", prompt)
    request_body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 500, "temperature": 0.7, "top_p": 0.9})
    logging.info("Raw request body: %s", request_body)
    
    try:
        response = bedrock_client.invoke_model(
            body=request_body,
            modelId='anthropic.claude-v2',
            accept='application/json',
            contentType='application/json'
        )
        if 'body' in response:
            raw_response_content = response['body'].read()
            logging.info("Raw response content: %s", raw_response_content)
            response_body = json.loads(raw_response_content)
            if 'completion' in response_body:
                return response_body['completion'].strip()
    except Exception as e:
        logging.exception("Exception during model invocation: %s", e)
        return "Error: Could not process the message."

# Streamlit UI for the app
st.title('Magic Compose: Enhance your writing with AI')
user_message = st.text_area("Enter your message:", "How are you?")
features = ['Change writing tone', 'Spelling and grammar', 'Shorten/elaborate', 'Translation', 'Expand My Writing', 'Analyze My Writing']
selected_feature = st.radio("Select a feature:", features)

# Initialize detail variable
detail = ""

# Handling different feature inputs
if selected_feature == 'Change writing tone':
    transformation_styles = ['Emojify', 'Make Formal', 'Make Polite', 'Shakespearify', 'Excited!', 'Chill', 'Lyrical', 'Custom']
    selected_style = st.radio("Choose your transformation style:", transformation_styles)
    detail = selected_style if selected_style != 'Custom' else st.text_input('Enter your custom style:')
elif selected_feature == 'Shorten/elaborate':
    detail = st.radio("Do you want to shorten or elaborate?", ('shorten', 'elaborate'))
elif selected_feature == 'Translation':
    detail = st.text_input("Enter target language (e.g., French, Spanish):")
elif selected_feature == 'Expand My Writing' or selected_feature == 'Analyze My Writing':
    detail = 'generate continuations' if selected_feature == 'Expand My Writing' else 'analyze text'  # Simplified handling for the example

# Button to initiate the process
if st.button('Apply Feature'):
    result_message = send_request_to_bedrock(user_message, selected_feature, detail)
    if result_message.startswith("Error:"):
        st.error(result_message)
    else:
        st.success("Here's the result:")
        st.write(result_message)

