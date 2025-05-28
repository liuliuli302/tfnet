'''
LLM Client Manager

This module provides a client to interact with various Large Language Models,
including those from OpenAI (e.g., GPT-4o), Google (e.g., Gemini Pro),
and placeholder for local models.

Setup:
1. Install necessary libraries:
   pip install openai google-generativeai

2. Set API Keys as environment variables:
   export OPENAI_API_KEY="your_openai_api_key"
   export GOOGLE_API_KEY="your_google_api_key"
'''
import os
import logging

# Attempt to import vendor libraries and provide guidance if they are missing
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI library not found. To use OpenAI models, please install it: pip install openai")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("Google GenerativeAI library not found. To use Gemini models, please install it: pip install google-generativeai")

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMClientManager:
    def __init__(self):
        self.openai_client = None
        self.gemini_model = None

        if OpenAI:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logging.info("OpenAI client initialized.")
            else:
                logging.warning("OPENAI_API_KEY environment variable not set. OpenAI models will not be available.")
        
        if genai:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                genai.configure(api_key=google_api_key)
                # For gemini-1.5-pro-latest or similar. Adjust model name as needed.
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                logging.info("Google Gemini client initialized with model 'gemini-1.5-pro-latest'.")
            else:
                logging.warning("GOOGLE_API_KEY environment variable not set. Gemini models will not be available.")

    def generate_text(self, prompt: str, model_name: str = "gpt-4o", **kwargs) -> str:
        '''
        Generates text using the specified model.

        Args:
            prompt (str): The input prompt for the LLM.
            model_name (str): The name of the model to use. 
                              Supported: "gpt-4o", "gemini-1.5-pro-latest", "local_placeholder".
            **kwargs: Additional keyword arguments for specific models.
                      For OpenAI: max_tokens, temperature, top_p, etc.
                      For Gemini: generation_config (e.g., genai.types.GenerationConfig(...))

        Returns:
            str: The generated text from the model, or an error message.
        '''
        if model_name.lower() == "gpt-4o":
            return self._generate_with_openai(prompt, model_name="gpt-4o", **kwargs)
        elif model_name.lower() == "gemini-1.5-pro-latest": # Ensure consistency with initialization
            return self._generate_with_gemini(prompt, **kwargs)
        elif model_name.lower() == "local_placeholder":
            return self._generate_with_local_model(prompt, **kwargs)
        else:
            error_msg = f"Model '{model_name}' is not supported. Supported models: gpt-4o, gemini-1.5-pro-latest, local_placeholder."
            logging.error(error_msg)
            return error_msg

    def _generate_with_openai(self, prompt: str, model_name: str = "gpt-4o", **kwargs) -> str:
        if not self.openai_client:
            return "OpenAI client is not initialized. Please check your API key."
        try:
            logging.info(f"Sending request to OpenAI model: {model_name}")
            # Default parameters if not provided in kwargs
            max_tokens = kwargs.pop('max_tokens', 1024)
            temperature = kwargs.pop('temperature', 0.7)
            
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs # Pass any other OpenAI specific params
            )
            logging.info("Received response from OpenAI.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return f"Error from OpenAI: {str(e)}"

    def _generate_with_gemini(self, prompt: str, **kwargs) -> str:
        if not self.gemini_model:
            return "Gemini client is not initialized. Please check your API key."
        try:
            logging.info("Sending request to Google Gemini model.")
            # Example of using generation_config if passed, otherwise defaults apply
            generation_config = kwargs.pop('generation_config', None)
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                **kwargs # Pass any other Gemini specific params
            )
            logging.info("Received response from Gemini.")
            # Handle potential lack of content or parts
            if response.parts:
                return ' '.join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text') and response.text: # Some responses might have .text directly
                 return response.text
            else:
                # Log the full response if it's not in the expected format
                logging.warning(f"Gemini response did not contain expected text parts. Full response: {response}")
                # Try to access candidate parts if available (common in some Gemini responses)
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    return ' '.join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                return "Gemini response did not contain text."

        except Exception as e:
            logging.error(f"Error calling Google Gemini API: {e}")
            return f"Error from Gemini: {str(e)}"

    def _generate_with_local_model(self, prompt: str, **kwargs) -> str:
        '''
        Placeholder for local model invocation.
        You would need to implement the logic to load and run your local model here.
        '''
        model_path = kwargs.get("model_path", "default/path/to/local_model")
        logging.info(f"Attempting to generate text with local model (placeholder) at {model_path} for prompt: '{prompt[:50]}...'")
        # Example: 
        # local_model = self.load_local_model(model_path) 
        # output = local_model.generate(prompt, **kwargs)
        # return output
        return f"Placeholder response for local model. Prompt: {prompt}"

# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Initializing LLM Client Manager...")
    manager = LLMClientManager()

    # Test OpenAI (ensure OPENAI_API_KEY is set)
    if manager.openai_client:
        print("\n--- Testing OpenAI GPT-4o ---")
        openai_prompt = "Explain the concept of zero-shot learning in simple terms."
        openai_response = manager.generate_text(openai_prompt, model_name="gpt-4o", max_tokens=150)
        print(f"Prompt: {openai_prompt}")
        print(f"GPT-4o Response: {openai_response}")
    else:
        print("\n--- OpenAI client not available (check API key) ---")

    # Test Gemini (ensure GOOGLE_API_KEY is set)
    if manager.gemini_model:
        print("\n--- Testing Google Gemini ---")
        gemini_prompt = "What are the key benefits of using a vector database?"
        # Example of passing specific config for Gemini if needed
        # config = genai.types.GenerationConfig(max_output_tokens=200, temperature=0.8)
        # gemini_response = manager.generate_text(gemini_prompt, model_name="gemini-1.5-pro-latest", generation_config=config)
        gemini_response = manager.generate_text(gemini_prompt, model_name="gemini-1.5-pro-latest")
        print(f"Prompt: {gemini_prompt}")
        print(f"Gemini Response: {gemini_response}")
    else:
        print("\n--- Gemini client not available (check API key) ---")

    # Test Local Model Placeholder
    print("\n--- Testing Local Model Placeholder ---")
    local_prompt = "Summarize the plot of 'War and Peace'."
    local_response = manager.generate_text(local_prompt, model_name="local_placeholder")
    print(f"Prompt: {local_prompt}")
    print(f"Local Model Response: {local_response}")

    print("\nLLM Client Manager example usage finished.")
