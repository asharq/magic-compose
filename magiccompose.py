import streamlit as st
import boto3
from botocore.config import Config
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the AWS Bedrock client with your AWS configuration
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',  # Adjust this to your AWS region
    config=Config(
        read_timeout=60,
        connect_timeout=60,
        retries={'max_attempts': 10, 'mode': 'adaptive'}
    )
)

def craft_prompt(message, style):
    # Define specific instructions for each style
    style_prompts = {
        'Emojify': f"Human: Please emojify the following message: '{message}'",
        'Make Formal': f"Human: Please make the following message more formal: '{message}'",
        'Make Polite': f"Human: Please make the following message more polite: '{message}'",
        'Shakespearify': f"Human: Please rewrite the following message in Shakespearean style: '{message}'",
        'Excited!': f"Human: Please rewrite the following message to sound more excited: '{message}'",
        'Chill': f"Human: Please make the following message sound more chill and relaxed: '{message}'",
        'Lyrical': f"Human: Please turn the following message into a short lyrical verse: '{message}'",
        # 'Custom' style handling is integrated within the function itself
    }

    # Select the prompt based on the chosen style or use a custom style prompt
    return style_prompts.get(style, f"Human: Please rewrite the following message to sound '{style}': '{message}'") + "\n\nAssistant:"

def transform_message_with_bedrock(message, style):
    """
    Transform a message using AWS Bedrock based on a selected style.
    """
    # Crafting the prompt according to the model's requirements using the new craft_prompt function
    complete_prompt = craft_prompt(message, style)

    # Print the input prompt for debugging
    logging.info("Sending prompt to model: %s", complete_prompt)

    # Configure the request to AWS Bedrock
    request_body = json.dumps({
        "prompt": complete_prompt,
        "max_tokens_to_sample": 100,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    # Print the raw request body being sent to the model
    logging.info("Raw request body: %s", request_body)

    # Invoke the AWS Bedrock model
    try:
        response = bedrock_client.invoke_model(
            body=request_body,
            modelId='anthropic.claude-v2',
            accept='application/json',
            contentType='application/json'
        )

        # Check if 'body' is in the response and read it
        if 'body' in response:
            raw_response_content = response['body'].read()

            # Print the raw response content from the model
            logging.info("Raw response content: %s", raw_response_content)

            response_body = json.loads(raw_response_content)

            # Print the parsed JSON response body
            logging.info("Parsed JSON response body: %s", response_body)

            if 'completion' in response_body:
                # Directly accessing the 'completion' value and stripping it for use
                transformed_message = response_body['completion'].strip()

                # Additional logging to confirm the value being returned
                logging.info("Transformed message being returned: %s", transformed_message)

                # Return the transformed message immediately
                return transformed_message

    except Exception as e:
        logging.exception("Exception during model invocation or response processing: %s", e)

    # Log right before returning the error message, indicating this path shouldn't be reached if transformation was successful
    logging.info("Returning error message due to some issue in processing.")
    return "Error: Could not transform the message."
   

# Streamlit UI for the app
st.title('Magic Compose for Stylus and Keyboard: Transform writing with Gen AI')

# User inputs the message they wish to transform
user_message = st.text_area("Enter your message:", "How are you?")

# Expanded list of user-selectable transformation styles with an option for custom style
transformation_styles = ['Emojify', 'Make Formal', 'Make Polite', 'Shakespearify', 'Excited!', 'Chill', 'Lyrical', 'Custom']
selected_style = st.radio("Choose your transformation style:", transformation_styles)

# Conditional display for custom style input
if selected_style == 'Custom':
    custom_style = st.text_input('Enter your custom style:')
    transformation_style = custom_style  # Use the custom style for transformation
else:
    transformation_style = selected_style  # Use the predefined style

# Button to initiate the transformation
if st.button('Transform Message'):
    transformed_message = transform_message_with_bedrock(user_message, transformation_style)
    
    # Check if the transformed message is not an error message before displaying
    if transformed_message.startswith("Error:"):
        st.error(transformed_message)
    else:
        st.write('Transformed Message:')
        st.success(transformed_message)
